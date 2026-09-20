"""
MedExplain AI - NLP extraction helpers used by Agent 2.
Module: medexplain.extraction

Deterministic extraction of patient metadata and structured test values.

Numbers are pulled out with table parsing and regular expressions, never by the
language model. That is a safety decision: a model misreading 8.2 as 3.2 fails
silently, whereas a pattern either matches or it does not. It also makes the
extraction reproducible, which the security assessments depend on.

Logic follows the working implementation in demo_presentation.py, with logging
in place of print() and type hints per AGENTS.md.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from medexplain.logging_config import audit, get_logger

logger = get_logger("MedExplain.Extraction")

DEFAULT_METADATA: Dict[str, str] = {
    "patient_name": "Unknown",
    "report_date": "Unknown",
    "panel_type": "Medical Lab Panel",
}

_METADATA_FIELDS = {
    "patient_name": [
        r"Patient Name:\s*([^|\n]+)",
        r"Patient:\s*([^|\n]+)",
        r"Name:\s*([^|\n]+)",
    ],
    "report_date": [
        r"Report Date:\s*([^|\n]+)",
        r"Collection Date:\s*([^|\n]+)",
        r"Date:\s*([^|\n]+)",
    ],
    "panel_type": [
        r"Panel Type:\s*([^|\n]+)",
        r"Panel:\s*([^|\n]+)",
        r"Test Name:\s*([^|\n]+)",
    ],
}

_LINE_PATTERNS: List[str] = [
    # 1. Name Value Unit Range Flag (e.g. Hemoglobin 11.5 g/dL 13.5-17.5 LOW)
    r"(?P<name>[A-Za-z0-9\s\(\)\-\/\_]{2,40})[\:\=\s]+"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[%a-zA-Z0-9\/\^\.\*\-]{1,15})\s*"
    r"(?:[\(\[\{]?(?P<ref>\d+(?:\.\d+)?\s*[\-\–\—\:]\s*\d+(?:\.\d+)?|[\<\>]\s*=?\s*\d+(?:\.\d+)?)[\]\}\)]?)\s*"
    r"(?P<flag>HIGH|LOW|NORMAL|ABNORMAL|H|L|N)",

    # 2. Name Value Unit Range (e.g. Hemoglobin 11.5 g/dL 13.5-17.5 or (13.5-17.5))
    r"(?P<name>[A-Za-z0-9\s\(\)\-\/\_]{2,40})[\:\=\s]+"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[%a-zA-Z0-9\/\^\.\*\-]{1,15})\s*"
    r"(?:[\(\[\{]?(?P<ref>\d+(?:\.\d+)?\s*[\-\–\—\:]\s*\d+(?:\.\d+)?|[\<\>]\s*=?\s*\d+(?:\.\d+)?)[\]\}\)]?)",

    # 3. Name Value Range Unit (e.g. Fasting Glucose 110 70-99 mg/dL)
    r"(?P<name>[A-Za-z0-9\s\(\)\-\/\_]{2,40})[\:\=\s]+"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?:[\(\[\{]?(?P<ref>\d+(?:\.\d+)?\s*[\-\–\—\:]\s*\d+(?:\.\d+)?|[\<\>]\s*=?\s*\d+(?:\.\d+)?)[\]\}\)]?)\s*"
    r"(?P<unit>[%a-zA-Z0-9\/\^\.\*\-]{1,15})",

    # 4. Name: Value Unit (e.g. WBC: 11.8 x10^3/uL)
    r"(?P<name>[A-Za-z0-9\s\(\)\-\/\_]{2,40}):\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>[%a-zA-Z0-9\/\^\.\*\-]{1,15})",

    # 5. Name  Value  Unit (tab or space separated)
    r"(?P<name>[A-Za-z0-9\s\(\)\-\/\_]{2,40})\s{2,}"
    r"(?P<val>\d+(?:\.\d+)?)\s+"
    r"(?P<unit>[%a-zA-Z0-9\/\^\.\*\-]{1,15})",
]

_HEADER_TOKENS = ("test description", "test name", "result", "reference", "units", "flag", "parameter", "analyte")

# --------------------------------------------------------------------------
# Analyte validation
# --------------------------------------------------------------------------

_NON_ANALYTE_TOKENS = {
    "accession", "address", "age", "barcode", "bed", "bill", "birth", "branch",
    "centre", "center", "client", "collected", "collection", "date", "dates", "datetime",
    "dob", "doctor", "email", "fax", "gender", "hospital", "id", "invoice", "lab no", "mobile", "mrn", "name",
    "no", "number", "num", "page", "pathologist", "patient", "phone", "physician", "printed", "received",
    "registered", "reported", "requisition", "sex", "signature", "specimen",
    "technician", "telephone", "uhid", "ward", "status", "comment", "notes",
}


_INVALID_UNITS = {
    "years", "year", "yrs", "yr", "months", "month", "mths", "days", "day",
    "hours", "hour", "hrs", "am", "pm", "male", "female", "m", "f",
}

_UNIT_RE = re.compile(r"^[A-Za-z%µμ][A-Za-z0-9%µμ/^.·\-]{0,14}$")
_RANGE_RE = re.compile(r"\d\s*(?:-|–|to)\s*\d|[<>]\s*=?\s*\d")

_MAX_NAME_LENGTH = 60


def _name_tokens(name: str) -> set:
    return set(re.findall(r"[a-z]+", str(name).lower()))


def _looks_like_unit(unit: str, test_name: str = "") -> bool:
    """True when the unit is plausibly a measurement unit."""
    text = str(unit or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in _INVALID_UNITS or lowered in _NON_ANALYTE_TOKENS:
        return False
    if test_name and (lowered == test_name.lower() or test_name.lower() in lowered or lowered in test_name.lower()):
        return False
    if len(text) > 12:
        return False
    if re.search(r"[%/^\dµμ]", text) or lowered in {"g/dl", "mg/dl", "meq/l", "u/l", "fl", "pg", "g/l", "iu/l", "ratio", "index"}:
        return True
    return bool(_UNIT_RE.match(text)) and len(text) <= 6



def _has_reference_range(reference_range: str) -> bool:
    return bool(_RANGE_RE.search(str(reference_range or "")))


def is_plausible_analyte(
    name: str,
    unit: str = "",
    reference_range: str = "",
    flag: str = "",
) -> Tuple[bool, str]:
    """
    Decide whether a parsed row is really a laboratory result.
    """
    text = str(name or "").strip()
    if not text:
        return False, "empty name"
    if len(text) > _MAX_NAME_LENGTH:
        return False, "name too long to be an analyte"
    if len(re.findall(r"[A-Za-z]", text)) < 2:
        return False, "name has no alphabetic content"

    overlap = _name_tokens(text) & _NON_ANALYTE_TOKENS
    if overlap:
        return False, f"administrative field ({', '.join(sorted(overlap))})"

    if not (
        _looks_like_unit(unit)
        or _has_reference_range(reference_range)
        or str(flag or "").upper() in {"HIGH", "LOW", "NORMAL", "ABNORMAL", "H", "L"}
    ):
        return False, "no unit, reference range or flag - not a measurement"

    return True, "ok"


def extract_metadata(raw_text: str) -> Dict[str, str]:
    """Pull patient metadata out of the report text with flexible regex patterns."""
    metadata = dict(DEFAULT_METADATA)
    flattened = " | ".join(line.strip() for line in raw_text.split("\n") if line.strip())

    for field, patterns in _METADATA_FIELDS.items():
        for pattern in patterns:
            match = re.search(pattern, flattened, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if val and val.lower() not in {"unknown", "n/a", "none"}:
                    metadata[field] = val
                    break

    logger.info(
        "Metadata: patient=%s | date=%s | panel=%s",
        metadata["patient_name"],
        metadata["report_date"],
        metadata["panel_type"],
    )
    return metadata


def _flag_from_range(value: float, reference_range: str) -> str:
    """Derive HIGH / LOW / NORMAL from a numeric reference range when no flag column exists."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", reference_range or "")
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        if value < low:
            return "LOW"
        if value > high:
            return "HIGH"
        return "NORMAL"

    less_match = re.search(r"<\s*(\d+(?:\.\d+)?)", reference_range or "")
    if less_match:
        limit = float(less_match.group(1))
        return "HIGH" if value > limit else "NORMAL"

    greater_match = re.search(r">\s*(\d+(?:\.\d+)?)", reference_range or "")
    if greater_match:
        limit = float(greater_match.group(1))
        return "LOW" if value < limit else "NORMAL"

    return "UNKNOWN"


def _clean_numeric_val(cell: str) -> Optional[float]:
    """Extract numeric float from cell string, ignoring flags/symbols."""
    text = str(cell or "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_from_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Structured extraction using PyMuPDF table detection with flexible cell parsing."""
    import pymupdf

    items: List[Dict[str, Any]] = []
    rejected: List[Tuple[str, str]] = []
    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.warning("Table extraction skipped, could not open %s: %s", pdf_path, exc)
        return items

    for page in document:
        try:
            tables = page.find_tables()
        except Exception:
            continue

        for table in tables:
            rows = table.extract()
            if not rows:
                continue

            for row in rows:
                if not row or len(row) < 2:
                    continue
                cells = [str(c).strip().replace("\n", " ") if c else "" for c in row]
                if any(token in cells[0].lower() for token in _HEADER_TOKENS):
                    continue

                # Find cell containing the numeric lab value
                val_idx = -1
                val_float = None
                for idx, cell in enumerate(cells):
                    # Check if cell is numeric or contains numeric value
                    fval = _clean_numeric_val(cell)
                    if fval is not None:
                        # Avoid treating dates like 2026 as lab values unless it's a 2+ column row with name
                        if idx > 0 or len(cells) > 2:
                            val_idx = idx
                            val_float = fval
                            break

                if val_idx == -1 or val_float is None:
                    continue

                # Test Name is typically in cells before val_idx (or cell 0 if val_idx is 1)
                name = cells[0] if val_idx > 0 else (cells[1] if len(cells) > 1 else "")
                if val_idx > 1 and len(cells[0]) < 3:  # handle code column e.g. "01 | HbA1c | 5.8"
                    name = cells[1]

                unit = ""
                reference = ""
                flag = ""

                # Look for unit, reference range, and flag in surrounding cells
                for idx, cell in enumerate(cells):
                    if idx == val_idx:
                        continue
                    if not unit and _looks_like_unit(cell, name):
                        unit = cell
                    elif not reference and _has_reference_range(cell):
                        reference = cell
                    elif not flag and cell.upper() in {"HIGH", "LOW", "NORMAL", "ABNORMAL", "H", "L", "N"}:
                        flag_upper = cell.upper()
                        if flag_upper in {"HIGH", "H", "ABNORMAL"}:
                            flag = "HIGH"
                        elif flag_upper in {"LOW", "L"}:
                            flag = "LOW"
                        elif flag_upper in {"NORMAL", "N"}:
                            flag = "NORMAL"

                resolved_flag = flag if flag in {"HIGH", "LOW", "NORMAL"} else _flag_from_range(val_float, reference)
                accepted, reason = is_plausible_analyte(name, unit, reference, resolved_flag)
                if not accepted:
                    rejected.append((name, reason))
                    continue

                items.append(
                    {
                        "test_name": name,
                        "value": val_float,
                        "unit": unit,
                        "reference_range": reference or "Not stated",
                        "flag": resolved_flag,
                    }
                )

    document.close()
    _log_rejected(rejected)
    return items


def _log_rejected(rejected: List[Tuple[str, str]]) -> None:
    """Record what the analyte filter dropped, and why."""
    if not rejected:
        return
    for name, reason in rejected[:20]:
        logger.info("  rejected %r - %s", name, reason)
    audit(
        "extraction.rows_rejected",
        count=len(rejected),
        samples=[{"name": n, "reason": r} for n, r in rejected[:10]],
    )
    logger.info("Filtered out %d non-analyte row(s)", len(rejected))


def _extract_from_text(raw_text: str) -> List[Dict[str, Any]]:
    """Regex fallback for reports without detectable tables."""
    items: List[Dict[str, Any]] = []
    rejected: List[Tuple[str, str]] = []

    for line in raw_text.split("\n"):
        stripped = line.strip()
        if not stripped or any(token in stripped.lower() for token in _HEADER_TOKENS):
            continue

        for pattern in _LINE_PATTERNS:
            match = re.search(pattern, stripped, re.IGNORECASE)
            if not match:
                continue

            groups = match.groupdict()
            try:
                value = float(groups["val"])
            except (ValueError, KeyError):
                continue

            reference = (groups.get("ref") or "").strip()
            explicit_flag = (groups.get("flag") or "").upper()
            flag = explicit_flag if explicit_flag in {"HIGH", "LOW", "NORMAL"} else _flag_from_range(value, reference)

            candidate_name = groups["name"].strip()
            candidate_unit = (groups.get("unit") or "").strip()
            accepted, reason = is_plausible_analyte(candidate_name, candidate_unit, reference, flag)
            if not accepted:
                rejected.append((candidate_name, reason))
                continue

            items.append(
                {
                    "test_name": candidate_name,
                    "value": value,
                    "unit": candidate_unit,
                    "reference_range": reference or "Not stated",
                    "flag": flag,
                }
            )
            break

    _log_rejected(rejected)
    return items


def _extract_from_llm(raw_text: str, pdf_path: str = "") -> List[Dict[str, Any]]:
    """LLM fallback extraction for complex/unstructured/scanned report text when regex and tables return 0 items."""
    import json
    import os
    from medexplain.llm import build_client

    client, _ = build_client(purpose="LLM fallback extraction")
    if not client:
        return []

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    prompt = (
        "You are an expert medical lab parser. Extract all laboratory test results from the document/text.\n"
        "Return ONLY a JSON array where each item has these exact keys:\n"
        '  "test_name": string (e.g. "Hemoglobin", "WBC", "Fasting Glucose", "Creatinine")\n'
        '  "value": number (numeric lab result float or integer)\n'
        '  "unit": string (e.g. "g/dL", "mg/dL", "%", "mEq/L", "x10^3/uL")\n'
        '  "reference_range": string (e.g. "13.5 - 17.5", "< 100", "70 - 99")\n'
        '  "flag": string ("HIGH", "LOW", "NORMAL", or "UNKNOWN")\n\n'
        "Format output inside ```json ``` markdown code blocks.\n"
    )

    contents: List[Any] = []
    pdf_bytes = None
    if pdf_path and os.path.exists(pdf_path):
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception:
            pdf_bytes = None

    if pdf_bytes and (len(raw_text.strip()) < 30 or "pymupdf" in raw_text.lower()):
        try:
            from google.genai import types
            part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            contents = [part, prompt]
        except Exception:
            contents = [f"{prompt}\n\nREPORT TEXT:\n{raw_text[:4000]}"]
    else:
        contents = [f"{prompt}\n\nREPORT TEXT:\n{raw_text[:4000]}"]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        text = (response.text or "").strip()
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        json_str = match.group(1) if match else text
        parsed = json.loads(json_str)

        items: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            for obj in parsed:
                if isinstance(obj, dict) and "test_name" in obj and "value" in obj:
                    name = str(obj.get("test_name", "")).strip()
                    try:
                        val = float(obj["value"])
                    except (ValueError, TypeError):
                        continue
                    unit = str(obj.get("unit", "")).strip()
                    ref = str(obj.get("reference_range", "")).strip()
                    flag = str(obj.get("flag", "UNKNOWN")).upper()
                    if flag not in {"HIGH", "LOW", "NORMAL"}:
                        flag = _flag_from_range(val, ref)

                    accepted, reason = is_plausible_analyte(name, unit, ref, flag)
                    if accepted:
                        items.append({
                            "test_name": name,
                            "value": val,
                            "unit": unit,
                            "reference_range": ref or "Not stated",
                            "flag": flag,
                        })
        if items:
            logger.info("LLM fallback successfully extracted %d test item(s)", len(items))
        return items
    except Exception as exc:
        logger.warning("LLM fallback extraction failed: %s", exc)
        return []


def extract_test_items(pdf_path: str, raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract structured test items.

    Tables are preferred because they preserve column semantics; the regex path
    is the fallback for text-only report formats, followed by LLM fallback.
    """
    items = _extract_from_tables(pdf_path)
    method = "table"

    if not items:
        items = _extract_from_text(raw_text)
        method = "regex"

    if not items:
        items = _extract_from_llm(raw_text, pdf_path)
        method = "llm-fallback"

    for item in items:
        logger.info(
            "  %-28s = %-10s %-10s (ref %s) [%s]",
            item["test_name"],
            item["value"],
            item["unit"],
            item["reference_range"],
            item["flag"],
        )

    logger.info("Extracted %d test item(s) via %s parsing", len(items), method)
    return items


