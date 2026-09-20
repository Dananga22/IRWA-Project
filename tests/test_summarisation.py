"""Tests for the summarisation module - the brief's second named NLP technique."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medexplain.summarisation import (  # noqa: E402
    split_sentences,
    summarise,
    summarise_findings,
    textrank_summary,
)

ITEMS = [
    {"test_name": "Total Cholesterol", "value": 245.0, "flag": "HIGH"},
    {"test_name": "HDL (Good Cholesterol)", "value": 38.0, "flag": "LOW"},
    {"test_name": "Platelets", "value": 240.0, "flag": "NORMAL"},
]

PASSAGES = [
    {
        "id": "p1",
        "title": "Cholesterol Levels",
        "source": "MedlinePlus",
        "source_url": "https://medlineplus.gov/lab-tests/cholesterol-levels/",
        "text": (
            "A cholesterol test measures total cholesterol, LDL cholesterol, HDL cholesterol "
            "and triglycerides. Cholesterol is a waxy fat-like substance the body needs in "
            "small amounts. The balance between the different types matters more than the total alone."
        ),
    },
    {
        "id": "p2",
        "title": "HDL Cholesterol",
        "source": "MedlinePlus",
        "source_url": "https://medlineplus.gov/lab-tests/cholesterol-levels/",
        "text": (
            "HDL is often called good cholesterol because it helps remove LDL cholesterol from "
            "the body. Higher HDL levels are associated with a lower chance of heart attack and stroke. "
            "Unlike LDL, a higher HDL result is generally regarded as the more favourable direction."
        ),
    },
    {
        "id": "p3",
        "title": "Platelet Tests",
        "source": "MedlinePlus",
        "source_url": "https://medlineplus.gov/lab-tests/platelet-tests/",
        "text": (
            "Platelets are small cell fragments that help blood clot and stop bleeding. "
            "A platelet count reports how many are in a given volume of blood. "
            "Counts are interpreted with the rest of the blood count and the person's history."
        ),
    },
]


class TestSentenceSplitting(unittest.TestCase):
    def test_decimals_do_not_split_sentences(self) -> None:
        text = "Your HbA1c is 8.2 percent. The range is 4.0 to 5.6 percent."
        self.assertEqual(len(split_sentences(text)), 2)

    def test_reference_ranges_survive(self) -> None:
        sentences = split_sentences("Hemoglobin sits between 13.5 - 17.5 g/dL normally.")
        self.assertIn("13.5 - 17.5", sentences[0])


class TestFindingsSummary(unittest.TestCase):
    def test_counts_are_correct(self) -> None:
        result = summarise_findings({"panel_type": "Lipid Panel"}, ITEMS)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["high"], 1)
        self.assertEqual(result["low"], 1)
        self.assertEqual(result["normal"], 1)

    def test_abnormal_tests_are_named(self) -> None:
        result = summarise_findings({}, ITEMS)
        self.assertIn("Total Cholesterol", result["text"])
        self.assertIn("HDL (Good Cholesterol)", result["text"])
        self.assertEqual(result["high_tests"], ["Total Cholesterol"])

    def test_all_normal_report_says_so(self) -> None:
        normal = [{"test_name": "Platelets", "value": 240, "flag": "NORMAL"}]
        self.assertIn("All of them fall within", summarise_findings({}, normal)["text"])

    def test_empty_report_does_not_crash(self) -> None:
        result = summarise_findings({}, [])
        self.assertEqual(result["total"], 0)
        self.assertTrue(result["text"])

    def test_summary_never_invents_a_test(self) -> None:
        """The summary is computed, so every named test must exist in the input."""
        result = summarise_findings({}, ITEMS)
        for name in result["high_tests"] + result["low_tests"]:
            self.assertIn(name, [i["test_name"] for i in ITEMS])


class TestTextRank(unittest.TestCase):
    def test_returns_at_most_top_n(self) -> None:
        self.assertLessEqual(len(textrank_summary(PASSAGES, top_n=3)), 3)

    def test_every_key_point_is_citable(self) -> None:
        for point in textrank_summary(PASSAGES, top_n=4):
            self.assertTrue(point["sentence"])
            self.assertTrue(point["title"])
            self.assertTrue(point["source_url"].startswith("http"))

    def test_sentences_are_extracted_not_generated(self) -> None:
        """Extractive summarisation must not invent text - every sentence is verbatim."""
        corpus = " ".join(p["text"] for p in PASSAGES)
        for point in textrank_summary(PASSAGES, top_n=4):
            self.assertIn(point["sentence"], corpus)

    def test_reading_order_is_preserved(self) -> None:
        points = textrank_summary(PASSAGES, top_n=4)
        corpus = " ".join(p["text"] for p in PASSAGES)
        positions = [corpus.index(p["sentence"]) for p in points]
        self.assertEqual(positions, sorted(positions))

    def test_query_terms_bias_selection(self) -> None:
        """Query-biased summarisation should favour the analytes actually tested."""
        biased = textrank_summary(PASSAGES, top_n=2, query_terms=["Platelets"])
        self.assertTrue(
            any("platelet" in p["sentence"].lower() for p in biased),
            "platelet sentences should surface when platelets were requested",
        )

    def test_near_duplicates_are_suppressed(self) -> None:
        duplicated = PASSAGES + [dict(PASSAGES[1], id="p2-copy")]
        points = textrank_summary(duplicated, top_n=4)
        self.assertEqual(len({p["sentence"] for p in points}), len(points))

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(textrank_summary([], top_n=4), [])

    def test_single_passage_still_works(self) -> None:
        self.assertTrue(textrank_summary([PASSAGES[0]], top_n=2))

    def test_short_fragments_are_ignored(self) -> None:
        noisy = [dict(PASSAGES[0], text="Yes. No. Maybe. " + PASSAGES[0]["text"])]
        for point in textrank_summary(noisy, top_n=3):
            self.assertGreaterEqual(len(point["sentence"].split()), 6)


class TestCombined(unittest.TestCase):
    def test_summarise_returns_both_outputs_and_names_its_method(self) -> None:
        result = summarise({"panel_type": "Lipid Panel"}, ITEMS, PASSAGES, top_n=3)
        self.assertTrue(result["findings"]["text"])
        self.assertTrue(result["key_points"])
        self.assertIn("TextRank", result["method"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
