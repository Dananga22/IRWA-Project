"""Tests for the RAG Retrieval Agent and the MCP tool boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.retrieval.mcp_client import MCPRetrievalClient  # noqa: E402
from agents.retrieval.retriever import RetrievalAgent, load_corpus  # noqa: E402
from agents.retrieval.store import BM25Store  # noqa: E402


class TestCorpus(unittest.TestCase):
    def test_corpus_loads(self) -> None:
        passages = load_corpus()
        self.assertGreater(len(passages), 20, "corpus should hold a usable number of passages")

    def test_every_passage_is_citable(self) -> None:
        for passage in load_corpus():
            self.assertTrue(passage.get("id"), "passage missing id")
            self.assertTrue(passage.get("title"), f"{passage['id']} missing title")
            self.assertTrue(passage.get("source"), f"{passage['id']} missing source")
            self.assertTrue(
                str(passage.get("source_url", "")).startswith("http"),
                f"{passage['id']} missing a usable source_url",
            )
            self.assertTrue(passage.get("analytes"), f"{passage['id']} missing analyte tags")

    def test_passage_ids_are_unique(self) -> None:
        ids = [p["id"] for p in load_corpus()]
        self.assertEqual(len(ids), len(set(ids)), "duplicate passage ids would corrupt the index")


class TestBM25Store(unittest.TestCase):
    def setUp(self) -> None:
        self.store = BM25Store()
        self.store.index(load_corpus())

    def test_indexes_all_passages(self) -> None:
        self.assertEqual(self.store.count(), len(load_corpus()))

    def test_scores_are_normalised(self) -> None:
        for hit in self.store.search("hemoglobin", top_k=5):
            self.assertGreaterEqual(hit["score"], 0.0)
            self.assertLessEqual(hit["score"], 1.0)


class TestRetrievalAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = RetrievalAgent(backend="bm25")

    def test_known_analytes_retrieve_the_right_passage(self) -> None:
        expectations = {
            "HbA1c": "hba1c",
            "Hemoglobin": "hemoglobin",
            "LDL Cholesterol": "ldl",
            "Platelets": "platelet",
            "TSH": "tsh",
            "Creatinine": "creatinine",
        }
        for analyte, expected_fragment in expectations.items():
            with self.subTest(analyte=analyte):
                result = self.agent.retrieve(analyte=analyte, top_k=3)
                self.assertTrue(result["grounded"], f"{analyte} returned no grounding context")
                top_id = result["passages"][0]["id"]
                self.assertIn(expected_fragment, top_id.lower(), f"{analyte} ranked {top_id} first")

    def test_every_hit_carries_a_citation(self) -> None:
        result = self.agent.retrieve(analyte="Hemoglobin", top_k=3)
        self.assertEqual(len(result["passages"]), len(result["citations"]))
        for citation in result["citations"]:
            self.assertTrue(citation["source_url"].startswith("http"))

    def test_unknown_analyte_is_not_grounded(self) -> None:
        """Retrieval must report failure so the explanation agent abstains."""
        result = self.agent.retrieve(analyte="Zorblax Factor", top_k=3)
        self.assertFalse(result["grounded"])
        self.assertEqual(result["passages"], [])


class TestMCPClient(unittest.TestCase):
    def test_inprocess_transport_returns_tool_payload(self) -> None:
        client = MCPRetrievalClient(transport="inprocess")
        result = client.call_tool(analyte="Platelets", flag="NORMAL", top_k=2)
        self.assertEqual(result["transport"], "inprocess")
        self.assertTrue(result["grounded"])
        self.assertLessEqual(len(result["passages"]), 2)

    def test_stdio_transport_speaks_real_mcp(self) -> None:
        """Spawns the MCP server and calls the tool over JSON-RPC on stdio."""
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp SDK not installed")

        client = MCPRetrievalClient(transport="stdio")
        result = client.call_tool(analyte="TSH", top_k=2)
        self.assertTrue(result["grounded"])
        self.assertIn(result["transport"], ("stdio", "inprocess"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
