"""Tests for the Localisation Agent (Agent 6) and its verification guards."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.translation.translator import (  # noqa: E402
    LANGUAGES,
    LocalisationAgent,
    supported_languages,
    verify_translation,
)

ITEMS = [
    {"test_name": "Total Cholesterol", "value": 245.0, "unit": "mg/dL", "reference_range": "125 - 200", "flag": "HIGH"},
    {"test_name": "HDL (Good Cholesterol)", "value": 38.0, "unit": "mg/dL", "reference_range": "> 40", "flag": "LOW"},
]

GOOD_SI = (
    "ඔබේ Total Cholesterol ප්‍රතිඵලය 245 mg/dL වන අතර එය 125 - 200 පරාසයට වඩා ඉහළය. "
    "ඔබේ HDL (Good Cholesterol) ප්‍රතිඵලය 38 mg/dL වේ. "
    "මෙය රෝග විනිශ්චයක් නොවේ. කරුණාකර ඔබේ වෛද්‍යවරයා හමුවන්න."
)


class TestLanguageRegistry(unittest.TestCase):
    def test_english_and_sinhala_are_offered(self) -> None:
        codes = {lang["code"] for lang in supported_languages()}
        self.assertIn("en", codes)
        self.assertIn("si", codes)

    def test_every_language_has_headings_and_a_native_name(self) -> None:
        for code, cfg in LANGUAGES.items():
            self.assertTrue(cfg.get("native_name"), f"{code} has no native name")
            for key in ("summary", "results", "questions", "notice"):
                self.assertTrue(cfg["headings"].get(key), f"{code} missing heading {key}")


class TestVerification(unittest.TestCase):
    """A translation is only shown if it survives all four checks."""

    def test_faithful_translation_passes(self) -> None:
        verdict = verify_translation(GOOD_SI, ITEMS, "si")
        self.assertTrue(verdict["ok"], verdict["issues"])

    def test_altered_measurement_is_rejected(self) -> None:
        """The most dangerous failure mode: a number changed in translation."""
        tampered = GOOD_SI.replace("245", "445")
        verdict = verify_translation(tampered, ITEMS, "si")
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["numeric_fidelity"])

    def test_dropped_measurement_is_rejected(self) -> None:
        partial = "ඔබේ HDL ප්‍රතිඵලය 38 mg/dL වේ. මෙය රෝග විනිශ්චයක් නොවේ."
        verdict = verify_translation(partial, ITEMS, "si")
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["numeric_fidelity"])

    def test_introduced_medication_is_rejected(self) -> None:
        unsafe = GOOD_SI + " ඔබ දිනකට atorvastatin 20 mg ගත යුතුය."
        verdict = verify_translation(unsafe, ITEMS, "si")
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["scope"])

    def test_dropped_disclaimer_is_rejected(self) -> None:
        without = GOOD_SI.split("මෙය රෝග")[0]
        verdict = verify_translation(without, ITEMS, "si")
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["disclaimer"])

    def test_untranslated_output_is_rejected(self) -> None:
        """Guards against the model echoing the English back unchanged."""
        english = "Your Total Cholesterol is 245 mg/dL and HDL is 38 mg/dL. Not a diagnosis."
        verdict = verify_translation(english, ITEMS, "si")
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["script"])

    def test_integer_and_decimal_forms_both_accepted(self) -> None:
        """245.0 written as 245 is a faithful rendering, not an alteration."""
        for written in ("245", "245.0"):
            text = GOOD_SI.replace("245", written)
            self.assertTrue(verify_translation(text, ITEMS, "si")["checks"]["numeric_fidelity"])


class _StubResponse:
    def __init__(self, payload: str) -> None:
        self.text = payload


class _StubModels:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def generate_content(self, **kwargs):
        return _StubResponse(self._payload)


class _StubClient:
    def __init__(self, payload: str) -> None:
        self.models = _StubModels(payload)


def _agent_returning(payload: str) -> LocalisationAgent:
    agent = LocalisationAgent(api_key="stub")
    agent._client = _StubClient(payload)  # only the network call is stubbed
    return agent


class TestLocalisationAgent(unittest.TestCase):
    PAYLOAD = json.dumps(
        {
            "summary": "සාරාංශයක්. ප්‍රතිඵල 2ක් ඇත.",
            "items": [
                {"test_name": "Total Cholesterol", "explanation": "ඔබේ ප්‍රතිඵලය 245 mg/dL වන අතර එය ඉහළය."},
                {"test_name": "HDL (Good Cholesterol)", "explanation": "ඔබේ ප්‍රතිඵලය 38 mg/dL වන අතර එය පහළය."},
            ],
            "questions": ["මෙය නැවත පරීක්ෂා කළ යුතුද?"],
            "notice": "මෙය රෝග විනිශ්චයක් නොවේ. ඔබේ වෛද්‍යවරයා හමුවන්න.",
        },
        ensure_ascii=False,
    )

    def test_english_is_returned_unchanged_without_a_model_call(self) -> None:
        agent = LocalisationAgent(api_key=None)
        result = agent.translate("Original English text.", ITEMS, "en")
        self.assertTrue(result.verified)
        self.assertEqual(result.explanation, "Original English text.")

    def test_unsupported_language_is_refused(self) -> None:
        result = _agent_returning(self.PAYLOAD).translate("text", ITEMS, "fr")
        self.assertFalse(result.available)

    def test_missing_key_reports_unavailable_in_the_target_language(self) -> None:
        agent = LocalisationAgent(api_key="your_gemini_api_key_here")
        result = agent.translate("English text", ITEMS, "si")
        self.assertFalse(result.available)
        self.assertIn("සිංහල", result.message)

    def test_successful_translation_is_composed_and_verified(self) -> None:
        result = _agent_returning(self.PAYLOAD).translate("English source", ITEMS, "si")
        self.assertTrue(result.verified, result.issues)
        self.assertIn("## සාරාංශය", result.explanation)
        self.assertIn("### Total Cholesterol", result.explanation, "test names must stay in English")
        self.assertEqual(set(result.items), {"Total Cholesterol", "HDL (Good Cholesterol)"})

    def test_translation_that_alters_a_value_is_withheld(self) -> None:
        tampered = self.PAYLOAD.replace("245", "999")
        result = _agent_returning(tampered).translate("English source", ITEMS, "si")
        self.assertFalse(result.verified)
        self.assertEqual(result.explanation, "", "an unverified translation must never be released")
        self.assertFalse(result.checks["numeric_fidelity"])

    def test_translation_that_adds_a_medication_is_withheld(self) -> None:
        unsafe = self.PAYLOAD.replace(
            "ඔබේ ප්‍රතිඵලය 245 mg/dL වන අතර එය ඉහළය.",
            "ඔබේ ප්‍රතිඵලය 245 mg/dL වේ. ඔබ atorvastatin ගත යුතුය.",
        )
        result = _agent_returning(unsafe).translate("English source", ITEMS, "si")
        self.assertFalse(result.verified)
        self.assertEqual(result.explanation, "")

    def test_malformed_model_output_degrades_gracefully(self) -> None:
        result = _agent_returning("not json at all").translate("English source", ITEMS, "si")
        self.assertFalse(result.verified)
        self.assertTrue(result.message)

    def test_markdown_fenced_json_is_tolerated(self) -> None:
        fenced = "```json\n" + self.PAYLOAD + "\n```"
        result = _agent_returning(fenced).translate("English source", ITEMS, "si")
        self.assertTrue(result.verified, result.issues)


if __name__ == "__main__":
    unittest.main(verbosity=2)
