"""
MedExplain AI — NLP Extraction Agent
Module: agents.extraction.extractor
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agents.extraction.patterns import ALIAS_MAP, LAB_LINE_REGEX, DEFAULT_REF_RANGES

logger = logging.getLogger("MedExplain.NLPExtractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


@dataclass
class LabValue:
    """Dataclass representing an extracted laboratory result biomarker."""
    test_name: str
    value: float
    unit: str
    reference_range: str
    flag: str  # HIGH | LOW | NORMAL | UNKNOWN
    raw_line: str
    confidence: float = 1.0


def normalize_test_name(raw_name: str) -> Tuple[str, str]:
    """Normalize raw test string into (canonical_key, display_name).

    Args:
        raw_name: Unsanitized test name string from report text.

    Returns:
        Tuple of (canonical_key, standardized_display_name).
    """
    clean_name = raw_name.strip().lower()
    # Remove leading non-alphanumeric chars
    clean_name = re.sub(r"^[^\w]+", "", clean_name)

    for alias, canonical in ALIAS_MAP.items():
        if alias in clean_name:
            # Format clean display title
            display_title = canonical.replace("_", " ").title()
            if canonical == "hba1c":
                display_title = "HbA1c"
            elif canonical == "tsh":
                display_title = "TSH"
            elif canonical == "alt":
                display_title = "ALT"
            elif canonical == "ast":
                display_title = "AST"
            elif canonical == "ldl_cholesterol":
                display_title = "LDL Cholesterol"
            elif canonical == "hdl_cholesterol":
                display_title = "HDL Cholesterol"
            return canonical, display_title

    return "unknown", raw_name.strip().title()


def parse_ref_range_bounds(ref_range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse low and high numerical bounds from reference range string.

    Args:
        ref_range_str: e.g. "13.5 - 17.5" or "< 100" or "70 - 99"

    Returns:
        Tuple of (low_bound, high_bound).
    """
    if not ref_range_str:
        return None, None

    # Handle "< 100" or "> 40"
    less_match = re.search(r"<\s*(\d+(?:\.\d+)?)", ref_range_str)
    if less_match:
        return 0.0, float(less_match.group(1))

    greater_match = re.search(r">\s*(\d+(?:\.\d+)?)", ref_range_str)
    if greater_match:
        return float(greater_match.group(1)), 99999.0

    # Range match: e.g. 13.5 - 17.5
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[\-\–\—\:]\s*(\d+(?:\.\d+)?)", ref_range_str)
    if range_match:
        return float(range_match.group(1)), float(range_match.group(2))

    return None, None


def determine_flag(value: float, canonical_name: str, explicit_flag: Optional[str], ref_range_str: str) -> str:
    """Determine HIGH, LOW, or NORMAL flag from value and reference range.

    Args:
        value: Numeric lab value.
        canonical_name: Normalized test key.
        explicit_flag: Flag string directly parsed from line (e.g. HIGH/H/L/NORMAL).
        ref_range_str: Reference range string.

    Returns:
        Flag string: 'HIGH', 'LOW', 'NORMAL', or 'UNKNOWN'.
    """
    if explicit_flag:
        flag_upper = explicit_flag.strip().upper()
        if flag_upper in ["HIGH", "H", "ABNORMAL"]:
            return "HIGH"
        elif flag_upper in ["LOW", "L"]:
            return "LOW"
        elif flag_upper in ["NORMAL", "N"]:
            return "NORMAL"

    low_b, high_b = parse_ref_range_bounds(ref_range_str)

    # Use default ranges if missing
    if low_b is None and high_b is None and canonical_name in DEFAULT_REF_RANGES:
        low_b, high_b = DEFAULT_REF_RANGES[canonical_name]

    if low_b is not None and high_b is not None:
        if value > high_b:
            return "HIGH"
        elif value < low_b:
            return "LOW"
        else:
            return "NORMAL"

    return "UNKNOWN"


def extract_values(text: str) -> List[LabValue]:
    """Extract laboratory test names, numerical values, units, reference ranges, and flags from raw text.

    Args:
        text: Raw text string extracted from PDF or lab report.

    Returns:
        List of LabValue dataclass objects.
    """
    results: List[LabValue] = []
    lines = text.splitlines()

    logger.info(f"Parsing {len(lines)} lines of text for lab values...")

    for line in lines:
        line_str = line.strip()
        if not line_str or len(line_str) < 5:
            continue

        match = LAB_LINE_REGEX.search(line_str)
        if match:
            raw_name = match.group("test_name")
            val_str = match.group("value")

            if not raw_name or not val_str:
                continue

            canonical, display_name = normalize_test_name(raw_name)
            if canonical == "unknown":
                continue  # Skip unmapped text lines

            try:
                val = float(val_str)
            except ValueError:
                continue

            unit = match.group("unit") or ""
            ref_range = match.group("ref_range") or ""
            explicit_flag = match.group("flag")

            # Fallback reference range string formatting if missing
            if not ref_range and canonical in DEFAULT_REF_RANGES:
                def_low, def_high = DEFAULT_REF_RANGES[canonical]
                ref_range = f"{def_low} - {def_high}"

            flag = determine_flag(val, canonical, explicit_flag, ref_range)

            lab_val = LabValue(
                test_name=display_name,
                value=val,
                unit=unit.strip(),
                reference_range=ref_range.strip(),
                flag=flag,
                raw_line=line_str,
                confidence=0.95
            )
            results.append(lab_val)
            logger.info(f"Extracted Lab Value: {display_name} = {val} {unit} [{flag}]")

    if not results:
        try:
            from medexplain.extraction import extract_test_items
            fallback_items = extract_test_items("", text)
            for item in fallback_items:
                display_name = str(item.get("test_name", "")).strip()
                val = float(item.get("value", 0.0))
                unit = str(item.get("unit", "")).strip()
                ref_range = str(item.get("reference_range", "")).strip()
                flag = str(item.get("flag", "UNKNOWN")).strip().upper()
                lab_val = LabValue(
                    test_name=display_name,
                    value=val,
                    unit=unit,
                    reference_range=ref_range,
                    flag=flag,
                    raw_line=f"{display_name} {val} {unit}",
                    confidence=0.95
                )
                results.append(lab_val)
                logger.info(f"Extracted Lab Value (via fallback): {display_name} = {val} {unit} [{flag}]")
        except Exception as exc:
            logger.warning(f"Fallback extraction failed: {exc}")

    return results
