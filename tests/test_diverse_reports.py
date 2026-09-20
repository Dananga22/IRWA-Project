"""
Unit tests for diverse PDF lab report extraction formats in MedExplain AI.
"""

import unittest
from medexplain.extraction import extract_test_items, _extract_from_text, extract_metadata
from agents.extraction.patterns import ALIAS_MAP, DEFAULT_REF_RANGES
from agents.extraction.extractor import normalize_test_name


class TestDiverseReportsExtraction(unittest.TestCase):

    def test_space_separated_lines_without_colons(self):
        raw_text = (
            "PATIENT REPORT\n"
            "Patient Name: Jane Doe\n"
            "Report Date: 2026-08-12\n\n"
            "TEST NAME                RESULT  UNIT       REFERENCE RANGE\n"
            "Hemoglobin               11.5    g/dL       13.5 - 17.5\n"
            "WBC                      12.5    x10^3/uL   4.5 - 11.0\n"
            "Fasting Blood Sugar      110.0   mg/dL      70 - 99\n"
            "Platelets                250     x10^3/uL   150 - 450\n"
        )
        items = _extract_from_text(raw_text)
        self.assertGreaterEqual(len(items), 3)

        names = [i["test_name"] for i in items]
        self.assertIn("Hemoglobin", names)
        self.assertIn("WBC", names)
        self.assertIn("Fasting Blood Sugar", names)

        # Verify derived flags
        hgb = next(i for i in items if i["test_name"] == "Hemoglobin")
        self.assertEqual(hgb["flag"], "LOW")

        wbc = next(i for i in items if i["test_name"] == "WBC")
        self.assertEqual(wbc["flag"], "HIGH")

    def test_lines_with_explicit_flags(self):
        raw_text = (
            "Serum Creatinine 1.4 mg/dL 0.7 - 1.3 HIGH\n"
            "Total Cholesterol 220 mg/dL 125 - 200 HIGH\n"
            "HDL 35 mg/dL > 40 LOW\n"
        )
        items = _extract_from_text(raw_text)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["flag"], "HIGH")

    def test_metadata_extraction_variations(self):
        raw_text = "Patient: John Smith\nCollection Date: 12/08/2026\nPanel: Lipid Panel"
        meta = extract_metadata(raw_text)
        self.assertEqual(meta["patient_name"], "John Smith")
        self.assertEqual(meta["report_date"], "12/08/2026")
        self.assertEqual(meta["panel_type"], "Lipid Panel")

    def test_nlp_extraction_normalization(self):
        canonical1, title1 = normalize_test_name("RBC")
        self.assertEqual(canonical1, "red_cell_count")
        canonical2, title2 = normalize_test_name("BUN")
        self.assertEqual(canonical2, "bun")


if __name__ == "__main__":
    unittest.main()

