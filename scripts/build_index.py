"""
MedExplain AI - Build the vector index for the Information Retrieval module.
Script: scripts/build_index.py

The index is built offline from knowledge_base/corpus.json and shipped as an
artifact, never built during a request. Re-run after editing the corpus.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --backend bm25
    python scripts/build_index.py --query "HbA1c"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.retrieval.retriever import RetrievalAgent, load_corpus  # noqa: E402
from medexplain.logging_config import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MedExplain knowledge base index.")
    parser.add_argument("--backend", choices=["auto", "chromadb", "bm25"], default="auto")
    parser.add_argument("--query", help="Run a test query after indexing")
    args = parser.parse_args()

    configure_logging()

    passages = load_corpus()
    print(f"Corpus contains {len(passages)} passages.")

    agent = RetrievalAgent(backend=args.backend)
    indexed = agent.reindex()

    print(f"Indexed {indexed} passages using backend: {agent.backend_name}")
    print(f"Index directory: {agent.index_dir}")

    sources = sorted({p.get("source", "?") for p in passages})
    print(f"Sources represented: {', '.join(sources)}")

    if args.query:
        print(f"\nTest query: {args.query!r}")
        result = agent.retrieve(analyte=args.query, top_k=3)
        if not result["passages"]:
            print("  No passages cleared the relevance threshold.")
        for passage in result["passages"]:
            print(f"  [{passage['score']:.3f}] {passage['title']}")
            print(f"          {passage['source']} - {passage['source_url']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
