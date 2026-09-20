"""
MedExplain AI - Vector store abstraction for the Information Retrieval module.
Module: agents.retrieval.store

Two backends, chosen automatically at runtime:

  1. ChromaDB (preferred, per AGENTS.md) with sentence-transformers embeddings
     if available, otherwise ChromaDB's bundled ONNX MiniLM embedder.
  2. A pure-Python BM25 index with no third-party dependencies.

The fallback exists so a live demonstration cannot fail because a machine is
missing a heavy dependency. Which backend is active is always logged and is
reported in the retrieval result, so nothing is hidden from the user.
"""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from medexplain.logging_config import get_logger

logger = get_logger("MedExplain.Retrieval.Store")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer shared by the BM25 backend."""
    return _TOKEN_RE.findall(text.lower())


class VectorStore(ABC):
    """Common interface for every retrieval backend."""

    backend_name: str = "abstract"

    @abstractmethod
    def index(self, passages: Sequence[Dict[str, Any]]) -> int:
        """Index passages. Returns the number indexed."""

    @abstractmethod
    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Return the top_k passages with a normalised `score` in [0, 1]."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed passages."""


class BM25Store(VectorStore):
    """
    Dependency-free BM25 ranking over the corpus.

    Used when ChromaDB is unavailable. BM25 is a standard lexical IR ranking
    function, so this backend is a genuine information-retrieval implementation
    rather than a stub.
    """

    backend_name = "bm25-inmemory"

    def __init__(self, persist_path: Optional[Path] = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.persist_path = persist_path
        self._passages: List[Dict[str, Any]] = []
        self._tokens: List[List[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_len: float = 0.0

    def index(self, passages: Sequence[Dict[str, Any]]) -> int:
        self._passages = list(passages)
        self._tokens = [tokenize(self._searchable_text(p)) for p in self._passages]
        self._doc_freq = Counter()
        for tokens in self._tokens:
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._avg_len = (sum(len(t) for t in self._tokens) / len(self._tokens)) if self._tokens else 0.0

        if self.persist_path is not None:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                self.persist_path.write_text(
                    json.dumps({"passages": self._passages}, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                logger.warning("Could not persist BM25 index to %s", self.persist_path)

        logger.info("BM25 index built over %d passages", len(self._passages))
        return len(self._passages)

    @staticmethod
    def _searchable_text(passage: Dict[str, Any]) -> str:
        analytes = " ".join(passage.get("analytes", []))
        return f"{passage.get('title', '')} {analytes} {analytes} {passage.get('text', '')}"

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self._passages:
            return []

        query_terms = tokenize(query)
        total_docs = len(self._passages)
        scored: List[Tuple[float, int]] = []

        for doc_id, tokens in enumerate(self._tokens):
            if not tokens:
                continue
            freqs = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                if term not in freqs:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self._avg_len or 1.0))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((score, doc_id))

        scored.sort(reverse=True)
        top = scored[:top_k]
        if not top:
            return []

        max_score = top[0][0] or 1.0
        results: List[Dict[str, Any]] = []
        for raw_score, doc_id in top:
            passage = dict(self._passages[doc_id])
            passage["score"] = round(min(raw_score / max_score, 1.0), 4)
            passage["raw_score"] = round(raw_score, 4)
            results.append(passage)
        return results

    def count(self) -> int:
        return len(self._passages)


class ChromaStore(VectorStore):
    """
    ChromaDB-backed dense retrieval.

    Uses sentence-transformers (all-MiniLM-L6-v2) when installed, otherwise
    ChromaDB's bundled ONNX MiniLM embedding function - both produce dense
    vector embeddings, so retrieval is semantic either way.
    """

    backend_name = "chromadb"

    def __init__(self, persist_dir: Path, collection_name: str = "medexplain_kb") -> None:
        import chromadb  # imported lazily so the fallback path needs no dependency

        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._embedding_fn, self.embedding_name = self._resolve_embedding_fn()

        # Probe the embedder before committing to this backend. Embedding models
        # are downloaded on first use, so an offline or firewalled machine only
        # fails here - and we want that failure now, not mid-demo.
        self._embedding_fn(["probe"])

        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as exc:
            if "mismatch" in str(exc).lower():
                logger.warning("Recreating ChromaDB collection due to embedding function mismatch: %s", exc)
                try:
                    self._client.delete_collection(name=collection_name)
                except Exception:
                    pass
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self._embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                raise
        self.backend_name = f"chromadb+{self.embedding_name}"

    @staticmethod
    def _resolve_embedding_fn() -> Tuple[Any, str]:
        from chromadb.utils import embedding_functions

        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        try:
            import sentence_transformers  # noqa: F401

            fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            fn(["probe"])
            logger.info("Embeddings: sentence-transformers/%s", model_name)
            return fn, f"sentence-transformers:{model_name}"
        except Exception:
            fn = embedding_functions.ONNXMiniLM_L6_V2()
            logger.info("Embeddings: ChromaDB ONNX MiniLM L6 v2 (dependency-free ONNX embedder)")
            return fn, "onnx-minilm"

    def index(self, passages: Sequence[Dict[str, Any]]) -> int:
        if not passages:
            return 0

        ids = [p["id"] for p in passages]
        documents = [
            f"{p.get('title', '')}. {' '.join(p.get('analytes', []))}. {p.get('text', '')}"
            for p in passages
        ]
        metadatas = [
            {
                "title": p.get("title", ""),
                "source": p.get("source", ""),
                "source_url": p.get("source_url", ""),
                "analytes": ", ".join(p.get("analytes", [])),
                "text": p.get("text", ""),
                "kind": p.get("kind", "explanation"),
                "section": p.get("section", ""),
            }
            for p in passages
        ]

        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("ChromaDB collection now holds %d passages", self._collection.count())
        return len(ids)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self._collection.count() == 0:
            return []

        response = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"],
        )

        results: List[Dict[str, Any]] = []
        ids = response.get("ids", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        for passage_id, meta, distance in zip(ids, metadatas, distances):
            # cosine distance -> similarity in [0, 1]
            similarity = max(0.0, 1.0 - float(distance))
            results.append(
                {
                    "id": passage_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "source_url": meta.get("source_url", ""),
                    "analytes": [a.strip() for a in str(meta.get("analytes", "")).split(",") if a.strip()],
                    "text": meta.get("text", ""),
                    "kind": meta.get("kind", "explanation"),
                    "section": meta.get("section", ""),
                    "score": round(similarity, 4),
                    "raw_score": round(float(distance), 4),
                }
            )
        return results

    def count(self) -> int:
        return self._collection.count()


def build_store(persist_dir: Path, prefer: Optional[str] = None) -> VectorStore:
    """
    Return the best available store.

    `prefer` may be "chromadb" or "bm25"; set VECTOR_BACKEND in the environment
    to force one. Falling back is logged loudly rather than done silently.
    """
    choice = (prefer or os.getenv("VECTOR_BACKEND", "auto")).lower()

    if choice in ("auto", "chromadb"):
        try:
            return ChromaStore(persist_dir=persist_dir)
        except Exception as exc:
            if choice == "chromadb":
                raise
            logger.warning("ChromaDB unavailable (%s). Falling back to BM25 retrieval backend.", exc)

    return BM25Store(persist_path=persist_dir / "bm25_index.json")
