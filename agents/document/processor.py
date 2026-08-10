"""
MedExplain AI — Document Processing Agent
Module: agents.document.processor
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

# Use PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz

logger = logging.getLogger("MedExplain.DocumentProcessor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class DocumentText:
    """Dataclass holding structured extracted text results from PDF documents."""
    raw_text: str
    page_count: int
    per_page: List[str]
    extraction_method: str = "PyMuPDF"
    likely_scanned: bool = False


class DocumentProcessingError(Exception):
    """Custom exception raised when PDF document validation or processing fails."""
    pass


def extract_text(pdf_path: str) -> DocumentText:
    """Extract raw text from a lab report PDF using PyMuPDF (fitz).

    Args:
        pdf_path: Absolute or relative file path to the PDF document.

    Returns:
        DocumentText dataclass containing raw text, per page breakdown, and scanned flag.

    Raises:
        DocumentProcessingError: If file is missing, invalid, encrypted, or over 10 MB.
    """
    if not os.path.exists(pdf_path):
        raise DocumentProcessingError(f"PDF file not found at path: '{pdf_path}'")

    file_size_bytes = os.path.getsize(pdf_path)
    max_size_bytes = 10 * 1024 * 1024  # 10 MB Limit

    if file_size_bytes > max_size_bytes:
        raise DocumentProcessingError(
            f"File size ({file_size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed limit of 10 MB."
        )

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise DocumentProcessingError(f"Invalid or corrupted PDF file: {e}")

    if doc.is_encrypted:
        raise DocumentProcessingError("PDF document is encrypted or password-protected and cannot be read.")

    page_count = len(doc)
    if page_count == 0:
        doc.close()
        raise DocumentProcessingError("PDF document contains 0 pages.")

    per_page: List[str] = []
    likely_scanned = False

    logger.info(f"Extracting text from PDF '{os.path.basename(pdf_path)}' ({page_count} pages)...")

    for idx in range(page_count):
        page = doc[idx]
        page_text = page.get_text("text").strip()
        per_page.append(page_text)

        # Check if page has low character yield (< 50 chars), indicating scanned image
        if len(page_text) < 50:
            likely_scanned = True
            logger.warning(f"Page {idx + 1} yielded only {len(page_text)} chars; marked likely_scanned=True.")

    doc.close()

    full_text = "\n\n".join(per_page)

    logger.info(f"Extraction complete! Total text length: {len(full_text)} chars. Scanned flag: {likely_scanned}")

    return DocumentText(
        raw_text=full_text,
        page_count=page_count,
        per_page=per_page,
        extraction_method="PyMuPDF",
        likely_scanned=likely_scanned
    )
