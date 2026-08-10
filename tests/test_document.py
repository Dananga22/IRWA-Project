"""
Unit tests for agents.document.processor module
File: tests/test_document.py
"""

import os
import unittest
import tempfile
try:
    import fitz
except ImportError:
    import pymupdf as fitz

from agents.document.processor import extract_text, DocumentText, DocumentProcessingError


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_pdf_path = os.path.join(self.temp_dir.name, "sample_report.pdf")

        # Create a valid test PDF using PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Complete Blood Count (CBC) Report\nPatient Name: John Doe\nHemoglobin: 14.2 g/dL (Normal: 13.5 - 17.5)\nWhite Cell Count: 7.4 x10^3/uL"
        )
        doc.save(self.sample_pdf_path)
        doc.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_text_valid_pdf(self):
        result = extract_text(self.sample_pdf_path)
        self.assertIsInstance(result, DocumentText)
        self.assertEqual(result.page_count, 1)
        self.assertIn("Hemoglobin: 14.2 g/dL", result.raw_text)
        self.assertFalse(result.likely_scanned)
        self.assertEqual(result.extraction_method, "PyMuPDF")

    def test_file_not_found(self):
        with self.assertRaises(DocumentProcessingError):
            extract_text("non_existent_file_xyz.pdf")

    def test_invalid_pdf_format(self):
        invalid_path = os.path.join(self.temp_dir.name, "bad.pdf")
        with open(invalid_path, "w") as f:
            f.write("This is not a PDF file content.")

        with self.assertRaises(DocumentProcessingError):
            extract_text(invalid_path)

    def test_likely_scanned_flag(self):
        scanned_pdf_path = os.path.join(self.temp_dir.name, "scanned_report.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Short")  # < 50 characters
        doc.save(scanned_pdf_path)
        doc.close()

        result = extract_text(scanned_pdf_path)
        self.assertTrue(result.likely_scanned)


if __name__ == "__main__":
    unittest.main()
