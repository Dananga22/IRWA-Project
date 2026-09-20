"""Tests for the durable approved-result store used by the localisation endpoint."""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medexplain import result_store  # noqa: E402

APPROVED_STATE = {
    "approved": True,
    "explanation": "## Summary\nYour cholesterol is above the stated range.",
    "test_items": [
        {"test_name": "Total Cholesterol", "value": 245.0, "unit": "mg/dL",
         "reference_range": "125 - 200", "flag": "HIGH"}
    ],
    "metadata": {"patient_name": "Nimal Perera", "report_date": "2026-08-05"},
}


class TestResultStore(unittest.TestCase):
    def setUp(self) -> None:
        result_store._MEMORY.clear()
        for path in result_store.STORE_DIR.glob("test*.json"):
            path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        result_store._MEMORY.clear()
        for path in result_store.STORE_DIR.glob("test*.json"):
            path.unlink(missing_ok=True)

    def test_approved_result_round_trips(self) -> None:
        self.assertTrue(result_store.save("abc123456789", APPROVED_STATE))
        loaded = result_store.load("abc123456789")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["explanation"], APPROVED_STATE["explanation"])

    def test_survives_a_process_restart(self) -> None:
        """The bug this store exists to fix: an in-memory cache lost on restart."""
        result_store.save("abc123456789", APPROVED_STATE)
        result_store._MEMORY.clear()  # simulate a fresh process
        self.assertIsNotNone(result_store.load("abc123456789"))

    def test_unapproved_results_are_never_stored(self) -> None:
        rejected = dict(APPROVED_STATE, approved=False)
        self.assertFalse(result_store.save("bbb111222333", rejected))
        self.assertIsNone(result_store.load("bbb111222333"))

    def test_patient_metadata_is_not_written_to_disk(self) -> None:
        """Data minimisation: only the values and the approved text are kept."""
        result_store.save("ccc111222333", APPROVED_STATE)
        raw = (result_store.STORE_DIR / "ccc111222333.json").read_text(encoding="utf-8")
        self.assertNotIn("Nimal Perera", raw)
        self.assertNotIn("metadata", json.loads(raw))
        self.assertIn("Total Cholesterol", raw)

    def test_malformed_trace_id_cannot_traverse_paths(self) -> None:
        self.assertIsNone(result_store.load("../../etc/passwd"))
        self.assertFalse(result_store.save("../../evil", APPROVED_STATE))

    def test_unknown_trace_id_returns_none(self) -> None:
        self.assertIsNone(result_store.load("ffffffffffff"))

    def test_expired_entries_are_dropped(self) -> None:
        result_store.save("ddd111222333", APPROVED_STATE)
        path = result_store.STORE_DIR / "ddd111222333.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["saved_at"] = time.time() - (result_store.RETENTION_SECONDS + 60)
        path.write_text(json.dumps(payload), encoding="utf-8")
        result_store._MEMORY.clear()

        self.assertIsNone(result_store.load("ddd111222333"))
        self.assertFalse(path.exists(), "an expired result should be deleted, not just hidden")


if __name__ == "__main__":
    unittest.main(verbosity=2)
