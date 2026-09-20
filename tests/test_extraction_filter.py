"""
Tests for the analyte filter in Agent 2.

Regression source: a real 13-page report where the header was parsed as results.
"Date Reported: 1/22/2025" became a test with value 1, and a patient's mobile
number became a test with value 77123456 - which was then retrieved against,
explained, and would have been sent to a third-party model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medexplain.extraction import _extract_from_text, is_plausible_analyte  # noqa: E402

REPORT = """Patient Name: Gayani Perera
Date Reported: 1/22/2025
Age: 34 Years
Sex: Female
Lab No: 88213
Mobile: 0771234567
Referring Doctor: Dr Silva
Page 1 of 13
Hemoglobin 13.5 g/dL 12.0 - 15.0 NORMAL
Total Cholesterol: 245 mg/dL
Prothrombin Time: 12.4 sec
Platelet Count: 240 x10^3/uL
"""


class TestAnalyteFilter(unittest.TestCase):
    def test_report_date_is_rejected(self) -> None:
        accepted, reason = is_plausible_analyte("Date Reported", "/22/2025", "Not stated", "")
        self.assertFalse(accepted)
        self.assertIn("administrative", reason)

    def test_phone_number_is_rejected(self) -> None:
        self.assertFalse(is_plausible_analyte("Mobile", "7", "Not stated", "")[0])

    def test_age_is_rejected_even_with_a_word_unit(self) -> None:
        self.assertFalse(is_plausible_analyte("Age", "Years", "Not stated", "")[0])

    def test_lab_number_is_rejected(self) -> None:
        accepted, reason = is_plausible_analyte("Lab No", "3", "Not stated", "")
        self.assertFalse(accepted)
        self.assertTrue("administrative" in reason or "not a measurement" in reason)

    def test_genuine_analytes_are_kept(self) -> None:
        for name, unit, ref in [
            ("Hemoglobin", "g/dL", "12.0 - 15.0"),
            ("Total Cholesterol", "mg/dL", "Not stated"),
            ("Platelet Count", "x10^3/uL", "150 - 450"),
        ]:
            with self.subTest(name=name):
                self.assertTrue(is_plausible_analyte(name, unit, ref, "")[0])

    def test_tests_whose_names_contain_time_or_count_survive(self) -> None:
        """The filter must not be so eager that it drops real tests."""
        self.assertTrue(is_plausible_analyte("Prothrombin Time", "sec", "11 - 13", "")[0])
        self.assertTrue(is_plausible_analyte("Absolute Neutrophil Count", "cells/uL", "1800 - 7800", "")[0])

    def test_flag_alone_is_enough_evidence(self) -> None:
        self.assertTrue(is_plausible_analyte("Some Analyte", "", "", "HIGH")[0])

    def test_end_to_end_on_a_realistic_header(self) -> None:
        names = [i["test_name"] for i in _extract_from_text(REPORT)]
        for junk in ("Date Reported", "Age", "Sex", "Lab No", "Mobile", "Page"):
            self.assertNotIn(junk, names, f"{junk} is not a laboratory analyte")
        self.assertIn("Hemoglobin", names)
        self.assertIn("Total Cholesterol", names)

    def test_no_patient_identifier_survives_extraction(self) -> None:
        """Privacy: an identifier parsed as a result would be sent onward as clinical data."""
        values = [i["value"] for i in _extract_from_text(REPORT)]
        self.assertNotIn(77123456.0, values, "a phone number must never become a test result")
        self.assertNotIn(8821.0, values, "a lab number must never become a test result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
