"""
MedExplain AI - Input security and privacy controls.
Module: medexplain.security

Three controls, applied before untrusted content reaches disk or the LLM:

  1. safe_upload_path   - blocks path traversal via a crafted upload filename.
  2. sanitize_document_text - neutralises prompt-injection payloads embedded in
     an uploaded PDF. Text inside a patient report is untrusted input, not
     instructions.
  3. redact_pii         - strips direct identifiers before any text is sent to
     a third-party LLM API.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from medexplain.logging_config import audit, get_logger
from agents.safety import rules

logger = get_logger("MedExplain.Security")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_SUFFIXES = {".pdf"}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

# Direct identifiers redacted before external API calls.
_PII_PATTERNS: List[Tuple[str, str]] = [
    (r"\b\d{9}[VvXx]\b", "[NIC]"),                                   # Sri Lankan old-format NIC
    (r"\b\d{12}\b", "[NIC]"),                                        # Sri Lankan new-format NIC
    (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[EMAIL]"),
    (r"(?:\+94|0)(?:\s|-)?\d{2}(?:\s|-)?\d{3}(?:\s|-)?\d{4}\b", "[PHONE]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"(?i)\b(?:patient\s*name|name)\s*[:\-]\s*[^\n|,]{2,60}", "Patient Name: [REDACTED]"),
    (r"(?i)\b(?:address)\s*[:\-]\s*[^\n|]{2,80}", "Address: [REDACTED]"),
    (r"(?i)\b(?:mrn|patient\s*id|hospital\s*no)\s*[:\-]\s*[A-Za-z0-9\-]{3,20}", "Patient ID: [REDACTED]"),
]


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def safe_upload_path(upload_dir: Path, original_filename: str) -> Path:
    """
    Return a safe destination path inside upload_dir.

    The original filename is never used as a path component. It is stripped to
    a safe basename, prefixed with a random token, and the result is verified to
    resolve inside upload_dir - so "../../etc/passwd" cannot escape.
    """
    if not original_filename:
        raise UploadValidationError("Missing filename.")

    normalised = unicodedata.normalize("NFKD", original_filename)
    basename = Path(normalised.replace("\\", "/")).name  # discard any directory part
    suffix = Path(basename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise UploadValidationError(f"Unsupported file type '{suffix or 'unknown'}'. Only PDF files are accepted.")

    stem = _SAFE_NAME_RE.sub("_", Path(basename).stem)[:80] or "report"
    destination = (upload_dir / f"{uuid.uuid4().hex[:8]}_{stem}{suffix}").resolve()

    upload_root = upload_dir.resolve()
    if upload_root not in destination.parents:
        audit("security.path_traversal_blocked", original_filename=original_filename)
        logger.error("Blocked path traversal attempt in upload filename: %r", original_filename)
        raise UploadValidationError("Invalid filename.")

    if basename != f"{Path(basename).stem}{suffix}" or ".." in original_filename or "/" in original_filename or "\\" in original_filename:
        audit("security.filename_sanitised", original_filename=original_filename, stored_as=destination.name)
        logger.warning("Sanitised suspicious upload filename %r -> %r", original_filename, destination.name)

    return destination


def validate_upload_size(size_bytes: int) -> None:
    """Reject oversized uploads."""
    if size_bytes > MAX_UPLOAD_BYTES:
        audit("security.upload_too_large", size_bytes=size_bytes, limit=MAX_UPLOAD_BYTES)
        raise UploadValidationError(
            f"File is {size_bytes / 1_048_576:.1f} MB. The maximum accepted size is {MAX_UPLOAD_BYTES // 1_048_576} MB."
        )


def sanitize_document_text(text: str) -> Dict[str, object]:
    """
    Neutralise prompt-injection payloads found in extracted PDF text.

    Matched instructions are replaced with a visible marker rather than removed
    silently, so an attempt is recorded in the audit log and stays visible to a
    reviewer - which is exactly what the prompt-injection assessment needs.
    """
    findings = rules.check_injection(text)
    cleaned = text

    if findings:
        for finding in findings:
            cleaned = re.sub(finding["pattern"], "[REMOVED: untrusted instruction]", cleaned, flags=re.IGNORECASE)
        audit(
            "security.prompt_injection_detected",
            count=len(findings),
            reasons=[f["reason"] for f in findings],
            samples=[f["matched_text"] for f in findings][:5],
        )
        logger.warning(
            "Prompt injection markers found in uploaded document: %s",
            ", ".join(sorted({f["reason"] for f in findings})),
        )

    return {"text": cleaned, "injection_findings": findings, "sanitised": bool(findings)}


def redact_pii(text: str) -> Dict[str, object]:
    """Strip direct identifiers before text is sent to a third-party LLM API."""
    redacted = text
    hits = 0
    for pattern, replacement in _PII_PATTERNS:
        redacted, count = re.subn(pattern, replacement, redacted)
        hits += count

    if hits:
        audit("security.pii_redacted", replacements=hits)
        logger.info("Redacted %d direct identifier(s) before external model call", hits)

    return {"text": redacted, "redactions": hits}
