"""Tests for the Care Guide Agent (Agent 7) and its verification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.care.guide import CareGuideAgent, check_care_guide, verify_care_guide  # noqa: E402
from agents.retrieval.retriever import get_retrieval_agent  # noqa: E402

TITLES = ["Healthy diet fact sheet", "Physical activity fact sheet"]

SAFE = {
    "eating": [{"text": "WHO describes a healthy diet as built around fruit, vegetables, legumes, "
                        "nuts and whole grains.", "source": "Healthy diet fact sheet"}],
    "activity": [{"text": "WHO's global recommendation for adults is at least 150 minutes of "
                          "moderate-intensity physical activity per week.",
                  "source": "Physical activity fact sheet"}],
}

ITEMS = [
    {"test_name": "LDL Cholesterol", "value": 162, "unit": "mg/dL", "reference_range": "< 100", "flag": "HIGH"},
    {"test_name": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "13.5-17.5", "flag": "LOW"},
]


class TestProhibitions(unittest.TestCase):
    """The guide informs; it must never prescribe."""

    def test_calorie_target_is_caught(self) -> None:
        self.assertTrue(check_care_guide("Aim for 1500 calories a day."))

    def test_supplement_is_caught(self) -> None:
        self.assertTrue(check_care_guide("Consider a vitamin D supplement each morning."))

    def test_supplement_dose_is_caught(self) -> None:
        self.assertTrue(check_care_guide("Take 1000 IU daily."))

    def test_meal_plan_is_caught(self) -> None:
        self.assertTrue(check_care_guide("For breakfast eat two eggs and oats."))

    def test_named_restrictive_diet_is_caught(self) -> None:
        self.assertTrue(check_care_guide("A keto diet may help here."))

    def test_imperative_instruction_is_caught(self) -> None:
        self.assertTrue(check_care_guide("You must stop eating rice."))

    def test_personal_target_is_caught(self) -> None:
        self.assertTrue(check_care_guide("Your target LDL should be under 70."))

    def test_published_population_figures_are_allowed(self) -> None:
        """WHO's own population guidance is citable; it is not a personal prescription."""
        self.assertEqual(
            check_care_guide(
                "WHO advises keeping salt intake below 5 g a day and at least 150 minutes of "
                "moderate activity per week, with at least 400 g of fruit and vegetables daily."
            ),
            [],
        )


class TestVerification(unittest.TestCase):
    def test_grounded_guide_passes(self) -> None:
        verdict = verify_care_guide(SAFE, TITLES)
        self.assertTrue(verdict["ok"], verdict["issues"])

    def test_uncited_statement_is_rejected(self) -> None:
        bad = {"eating": [{"text": "Eat more oily fish.", "source": "Some Blog"}]}
        verdict = verify_care_guide(bad, TITLES)
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["every_claim_cited"])

    def test_medication_is_rejected(self) -> None:
        bad = {"habits": [{"text": "Ask about starting atorvastatin.", "source": TITLES[0]}]}
        verdict = verify_care_guide(bad, TITLES)
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["scope"])

    def test_diagnosis_is_rejected(self) -> None:
        bad = {"habits": [{"text": "Based on this you have diabetes.", "source": TITLES[0]}]}
        self.assertFalse(verify_care_guide(bad, TITLES)["ok"])

    def test_empty_guide_is_not_ok(self) -> None:
        self.assertFalse(verify_care_guide({"eating": []}, TITLES)["checks"]["has_content"])


class TestCorpusSeparation(unittest.TestCase):
    """
    Guidance and explanation passages must never mix. Lifestyle content inside a
    clinical explanation would read as advice the explanation agent may not give.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = get_retrieval_agent()

    def test_explanation_retrieval_returns_no_guidance(self) -> None:
        for analyte in ["LDL Cholesterol", "Hemoglobin", "HbA1c", "Sodium"]:
            with self.subTest(analyte=analyte):
                result = self.agent.retrieve(analyte=analyte, top_k=3, kind="explanation")
                for passage in result["passages"]:
                    self.assertNotEqual(passage.get("kind"), "guidance")

    def test_guidance_retrieval_returns_no_explanations(self) -> None:
        for analyte in ["LDL Cholesterol", "HbA1c"]:
            with self.subTest(analyte=analyte):
                result = self.agent.retrieve(analyte=analyte, top_k=3, kind="guidance")
                for passage in result["passages"]:
                    self.assertEqual(passage.get("kind"), "guidance")


class TestAgent(unittest.TestCase):
    def test_builds_a_verified_guide_without_an_api_key(self) -> None:
        """The deterministic path must work, so a demo cannot fail on the API."""
        agent = CareGuideAgent(api_key=None)
        guide = agent.build(ITEMS)
        self.assertTrue(guide.verified, guide.issues)
        self.assertTrue(any(guide.sections.values()))
        self.assertTrue(guide.citations)

    def test_every_statement_carries_a_source(self) -> None:
        guide = CareGuideAgent(api_key=None).build(ITEMS)
        for entries in guide.sections.values():
            for entry in entries:
                self.assertTrue(entry["source"], "a care guide statement must cite its source")

    def test_output_contains_no_prescription(self) -> None:
        guide = CareGuideAgent(api_key=None).build(ITEMS)
        combined = " ".join(e["text"] for entries in guide.sections.values() for e in entries)
        self.assertEqual(check_care_guide(combined), [])

    def test_notice_is_always_present(self) -> None:
        guide = CareGuideAgent(api_key=None).build(ITEMS)
        self.assertIn("not a diagnosis", guide.notice.lower())
        self.assertIn("doctor", guide.notice.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
