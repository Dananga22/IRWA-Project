"""
MedExplain AI - RAG Retrieval Agent (Information Retrieval module).
Module: agents.retrieval.retriever

Agent 3 of the pipeline. Searches the curated WHO / CDC / MedlinePlus corpus
and returns passages together with their source title and URL, so that every
claim in the final explanation can be cited and verified.

Exposed to the LLM Explanation Agent as an MCP tool - see
agents/retrieval/mcp_server.py and agents/retrieval/mcp_client.py.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from medexplain.logging_config import audit, get_logger
from agents.retrieval.store import VectorStore, build_store

logger = get_logger("MedExplain.Retrieval.Agent")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "knowledge_base" / "corpus.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "knowledge_base" / "index"

# Below this similarity the agent reports no usable context, so the explanation
# agent abstains instead of the model answering from memory.
#
# The threshold has to differ by backend because the score distributions do.
# BM25 scores are normalised so the best hit is 1.0 and unrelated text lands
# near zero. Dense cosine similarity is compressed: MiniLM puts two unrelated
# medical sentences around 0.3, so a 0.15 threshold marks everything "grounded"
# - a report containing analytes absent from the corpus was reporting 69/69
# grounded and citing irrelevant sources for all of them.
MIN_RELEVANCE_SPARSE = float(os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.15"))
MIN_RELEVANCE_DENSE = float(os.getenv("RETRIEVAL_MIN_RELEVANCE_DENSE", "0.40"))


def min_relevance_for(backend_name: str) -> float:
    """Pick the relevance floor appropriate to the active retrieval backend."""
    return MIN_RELEVANCE_SPARSE if "bm25" in (backend_name or "").lower() else MIN_RELEVANCE_DENSE


class RetrievalError(Exception):
    """Raised when the knowledge base cannot be loaded or searched."""


def load_corpus(corpus_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load the curated corpus from disk."""
    path = corpus_path or DEFAULT_CORPUS
    if not path.exists():
        raise RetrievalError(f"Knowledge base not found at {path}. Run scripts/build_index.py first.")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Could not read knowledge base at {path}: {exc}") from exc

    passages = data.get("passages", [])
    if not passages:
        raise RetrievalError(f"Knowledge base at {path} contains no passages.")
    return passages


class RetrievalAgent:
    """
    [AGENT 3: RAG Retrieval Agent]

    Task: given a lab analyte (and optionally its value and flag), retrieve the
    most relevant passages from the trusted medical corpus, with citations.
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.corpus_path = corpus_path or DEFAULT_CORPUS
        self.index_dir = index_dir or DEFAULT_INDEX_DIR
        self.store: VectorStore = build_store(self.index_dir, prefer=backend)
        self._ensure_indexed()

    @property
    def backend_name(self) -> str:
        return self.store.backend_name

    def _ensure_indexed(self) -> None:
        """Index the corpus if the store is empty."""
        if self.store.count() > 0:
            logger.info(
                "Retrieval agent ready | backend=%s | passages=%d",
                self.store.backend_name,
                self.store.count(),
            )
            return

        passages = load_corpus(self.corpus_path)
        try:
            indexed = self.store.index(passages)
        except Exception as exc:
            # Indexing can fail after construction (e.g. an embedding model that
            # only downloads on first real use). Fall back rather than abort.
            logger.warning(
                "Indexing failed on backend %s (%s). Falling back to BM25 retrieval backend.",
                self.store.backend_name,
                exc,
            )
            from agents.retrieval.store import BM25Store

            self.store = BM25Store(persist_path=self.index_dir / "bm25_index.json")
            indexed = self.store.index(passages)

        logger.info(
            "Retrieval agent indexed corpus | backend=%s | passages=%d",
            self.store.backend_name,
            indexed,
        )

    def reindex(self) -> int:
        """Force a rebuild of the index from the corpus file."""
        passages = load_corpus(self.corpus_path)
        return self.store.index(passages)

    def retrieve(
        self,
        analyte: str,
        value: Optional[Any] = None,
        flag: Optional[str] = None,
        top_k: int = 3,
        kind: str = "explanation",
    ) -> Dict[str, Any]:
        """
        Retrieve grounding context for one analyte.

        Returns a dict with `passages`, `citations`, `grounded` and `backend`.
        `grounded` is False when nothing clears MIN_RELEVANCE, which instructs
        the explanation agent to abstain rather than generate from memory.

        `kind` selects which half of the corpus is searched. "explanation"
        passages describe what a test measures; "guidance" passages carry general
        lifestyle information. They are kept strictly apart: lifestyle content
        appearing inside a clinical explanation would read as advice the
        explanation agent is not permitted to give, and would trip the safety
        scope rules.
        """
        # The analyte name is repeated so it dominates lexical scoring; generic
        # filler words would match every passage and dilute the ranking.
        query_parts = [analyte, analyte]
        if flag and str(flag).upper() in {"HIGH", "LOW"}:
            query_parts.append("above the reference range" if str(flag).upper() == "HIGH" else "below the reference range")
        query = " ".join(str(p) for p in query_parts)

        try:
            # Over-fetch generously, filter to the requested half of the corpus,
            # then re-rank by analyte match before truncating.
            hits = self.store.search(query, top_k=max(top_k * 8, 24))
            hits = [h for h in hits if (h.get("kind") or "explanation") == kind]
            hits = self._rerank_by_analyte(analyte, hits)[:top_k]
        except Exception as exc:  # a retrieval failure must not crash the pipeline
            logger.error("Retrieval failed for analyte %r: %s", analyte, exc)
            audit("retrieval.error", analyte=analyte, error=str(exc))
            return {
                "analyte": analyte,
                "query": query,
                "kind": kind,
                "passages": [],
                "citations": [],
                "grounded": False,
                "backend": self.store.backend_name,
                "error": str(exc),
            }

        threshold = min_relevance_for(self.store.backend_name)
        relevant = [h for h in hits if h.get("score", 0.0) >= threshold]
        citations = [
            {"title": h["title"], "source": h["source"], "source_url": h["source_url"], "passage_id": h["id"]}
            for h in relevant
        ]

        audit(
            "agent.retrieval",
            analyte=analyte,
            query=query,
            backend=self.store.backend_name,
            kind=kind,
            hits=len(hits),
            kept=len(relevant),
            threshold=threshold,
            passage_ids=[h["id"] for h in relevant],
            top_score=relevant[0]["score"] if relevant else 0.0,
        )
        logger.info(
            "Retrieved %d/%d passages for %r (top score %.3f, threshold %.2f)",
            len(relevant),
            len(hits),
            analyte,
            hits[0]["score"] if hits else 0.0,
            threshold,
        )
        if hits and not relevant:
            logger.info(
                "  %r is not covered by the knowledge base - the explanation agent will abstain",
                analyte,
            )

        return {
            "analyte": analyte,
            "query": query,
            "kind": kind,
            "passages": relevant,
            "citations": citations,
            "grounded": bool(relevant),
            "backend": self.store.backend_name,
        }

    @staticmethod
    def _rerank_by_analyte(analyte: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Boost passages whose declared analytes match the requested test name.

        Backend-agnostic: the corpus tags each passage with the analyte aliases
        it covers, so an exact or partial alias match is strong evidence of
        topical relevance that a generic similarity score can miss.
        """
        wanted = analyte.strip().lower()
        wanted_tokens = {t for t in re.split(r"[^a-z0-9]+", wanted) if len(t) > 2}

        reranked: List[Dict[str, Any]] = []
        for hit in hits:
            aliases = [str(a).strip().lower() for a in hit.get("analytes", [])]
            if any(wanted == alias for alias in aliases):
                boost = 3.0          # exact alias match
            elif any(wanted in alias or alias in wanted for alias in aliases if alias):
                boost = 1.5          # one name contains the other
            elif wanted_tokens and any(wanted_tokens & {t for t in re.split(r"[^a-z0-9]+", alias) if t} for alias in aliases):
                boost = 1.2          # shared word
            else:
                boost = 1.0

            scored = dict(hit)
            scored["match_boost"] = boost
            # Rank on the uncapped product - clamping before sorting would
            # collapse strong and weak matches to the same value.
            scored["rank_score"] = hit.get("score", 0.0) * boost
            scored["score"] = round(min(scored["rank_score"], 1.0), 4)
            reranked.append(scored)

        reranked.sort(key=lambda h: h["rank_score"], reverse=True)
        return reranked

    def retrieve_for_items(self, test_items: Sequence[Dict[str, Any]], top_k: int = 3) -> Dict[str, Dict[str, Any]]:
        """Retrieve context for every extracted test item, keyed by test name."""
        context: Dict[str, Dict[str, Any]] = {}
        for item in test_items:
            name = str(item.get("test_name", "")).strip()
            if not name:
                continue
            context[name] = self.retrieve(analyte=name, value=item.get("value"), flag=item.get("flag"), top_k=top_k)
        return context


_AGENT_SINGLETON: Optional[RetrievalAgent] = None


def get_retrieval_agent() -> RetrievalAgent:
    """Process-wide singleton so the index is loaded once, not per request."""
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = RetrievalAgent()
    return _AGENT_SINGLETON
