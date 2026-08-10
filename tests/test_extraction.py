"""
Unit tests for agents.extraction.extractor module
File: tests/test_extraction.py
"""

import unittest
from agents.extraction.extractor import extract_values, LabValue, normalize_test_name, parse_ref_range_bounds


class TestNLPExtractor(unittest.TestCase):

    def test_normalize_test_name(self):
        self.assertEqual(normalize_test_name("HbA1c")[0], "hba1c")
        self.assertEqual(normalize_test_name("Hemoglobin A1c")[0], "hba1c")
        self.assertEqual(normalize_test_name("Fasting Blood Sugar")[0], "fasting_glucose")
        self.assertEqual(normalize_test_name("FBS")[0], "fasting_glucose")
        self.assertEqual(normalize_test_name("HGB")[0], "haemoglobin")
        self.assertEqual(normalize_test_name("Serum Creatinine")[0], "creatinine")
        self.assertEqual(normalize_test_name("SGPT")[0], "alt")
        self.assertEqual(normalize_test_name("SGOT")[0], "ast")
        self.assertEqual(normalize_test_name("Leukocytes")[0], "white_cell_count")

    def test_parse_ref_range_bounds(self):
        low, high = parse_ref_range_bounds("13.5 - 17.5")
        self.assertEqual(low, 13.5)
        self.assertEqual(high, 17.5)

        low, high = parse_ref_range_bounds("< 100")
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 100.0)

    def test_15_realistic_report_lines(self):
        sample_report_text = """
1. HbA1c: 8.2% (4.0 - 5.6) HIGH
2. Fasting Glucose -- 145 mg/dL [70 - 99] HIGH
3. Hemoglobin: 11.2 g/dL (13.5 - 17.5) LOW
4. Total Cholesterol: 220 mg/dL (125 - 199) HIGH
5. LDL Cholesterol  142 mg/dL  (0 - 99)  HIGH
6. HDL Cholesterol: 48 mg/dL (40 - 60) NORMAL
7. Triglycerides: 185 mg/dL (0 - 149) ABNORMAL
8. Serum Creatinine: 1.1 mg/dL (0.74 - 1.35)
9. ALT (SGPT) : 65 U/L (7 - 56) H
10. AST (SGOT): 28 U/L (10 - 40) NORMAL
11. TSH: 5.2 mIU/L (0.4 - 4.0) HIGH
12. 25-Hydroxy Vitamin D: 18.5 ng/mL (30.0 - 100.0) LOW
13. White Cell Count: 12.5 x10^3/uL (4.5 - 11.0) HIGH
14. HGB : 14.0 g/dL (13.5 - 17.5)
15. Vit D : 15.0 ng/mL
"""
        values = extract_values(sample_report_text)
        self.assertGreaterEqual(len(values), 15)

        # Verify specific extractions
        hba1c_item = next(v for v in values if v.test_name == "HbA1c")
        self.assertEqual(hba1c_item.value, 8.2)
        self.assertEqual(hba1c_item.flag, "HIGH")

        fbs_item = next(v for v in values if v.test_name == "Fasting Glucose")
        self.assertEqual(fbs_item.value, 145.0)
        self.assertEqual(fbs_item.flag, "HIGH")

        hb_item = next(v for v in values if v.test_name == "Haemoglobin")
        self.assertEqual(hb_item.value, 11.2)
        self.assertEqual(hb_item.flag, "LOW")

        wbc_item = next(v for v in values if v.test_name == "White Cell Count")
        self.assertEqual(wbc_item.value, 12.5)
        self.assertEqual(wbc_item.flag, "HIGH")

if __name__ == "__main__":
    unittest.main()
