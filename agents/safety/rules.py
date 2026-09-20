"""
MedExplain AI - Deterministic rule set for the Safety Verification Agent.
Module: agents.safety.rules

These rules are intentionally deterministic. A model-based check alone can be
talked around; a regex either matches or it does not, which is what makes the
safety gate testable and reproducible for the individual security assessments.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Scope violations - the system explains, it does not diagnose or prescribe.
# --------------------------------------------------------------------------

DIAGNOSIS_PATTERNS: List[Tuple[str, str]] = [
    (r"\byou (?:have|are suffering from|are diagnosed with)\b", "states a diagnosis"),
    (r"\bthis (?:means|confirms|indicates) (?:that )?you have\b", "states a diagnosis"),
    (r"\byou (?:definitely|certainly|clearly) have\b", "states a diagnosis"),
    (r"\bdiagnos(?:is|ed|e) (?:is|as|with)\b", "states a diagnosis"),
    (r"\byou are (?:diabetic|anaemic|anemic|hypertensive)\b", "states a diagnosis"),
]

PRESCRIPTION_PATTERNS: List[Tuple[str, str]] = [
    (r"\byou should (?:take|start|use)\b.{0,40}\b(?:mg|mcg|tablet|capsule|dose|doses|injection)\b", "recommends a medication"),
    (r"\b(?:take|start|begin)\s+\d+\s*(?:mg|mcg|ml|g)\b", "gives a dosage"),
    (r"\b(?:metformin|insulin|atorvastatin|simvastatin|statins?|aspirin|levothyroxine|ibuprofen|paracetamol|amoxicillin|warfarin|prednisolone)\b", "names a medication"),
    (r"\bi (?:recommend|prescribe|suggest) (?:that you )?(?:take|start)\b", "recommends a medication"),
    (r"\byou (?:should |can |may )?stop taking\b", "advises altering treatment"),
    (r"\b(?:increase|decrease|reduce|double|halve) your (?:dose|dosage|medication)\b", "advises altering treatment"),
]

PROGNOSIS_PATTERNS: List[Tuple[str, str]] = [
    (r"\byou will (?:develop|get|suffer|die)\b", "predicts prognosis"),
    (r"\bwithin \d+ (?:months?|years?) you will\b", "predicts prognosis"),
    (r"\blife expectancy\b", "predicts prognosis"),
    (r"\bthere is no (?:hope|cure for you)\b", "predicts prognosis"),
]

# --------------------------------------------------------------------------
# Tone - alarming or dismissive language is unsafe in a patient-facing tool.
# --------------------------------------------------------------------------

ALARMING_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(?:fatal|deadly|terminal|life-threatening|dying)\b", "alarming language"),
    (r"\byou (?:could|will|might) die\b", "alarming language"),
    (r"\b(?:cancer|tumou?r|malignan\w+)\b", "names a serious condition speculatively"),
]

DISMISSIVE_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(?:nothing to worry about at all|completely fine, ignore)\b", "dismissive language"),
    (r"\bno need to (?:see|consult|contact) (?:a |your )?doctor\b", "discourages clinical consultation"),
    (r"\byou (?:don't|do not) need (?:a |to see a )?doctor\b", "discourages clinical consultation"),
]

# --------------------------------------------------------------------------
# Prompt injection - text arriving from an uploaded PDF is untrusted input.
# --------------------------------------------------------------------------

INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"ignore (?:all |any )?(?:the )?(?:previous|prior|above) instructions?", "instruction override attempt"),
    (r"disregard (?:all |any )?(?:the )?(?:previous|prior|above|system)", "instruction override attempt"),
    (r"\byou are now\b.{0,40}\b(?:doctor|physician|unrestricted|dan|jailbroken)\b", "role reassignment attempt"),
    (r"\b(?:system prompt|system instruction)s?\b", "prompt leakage attempt"),
    (r"\brepeat (?:your|the) (?:instructions|prompt|rules)\b", "prompt leakage attempt"),
    (r"\bact as (?:if you are )?(?:a |an )?(?:doctor|physician|medical professional)\b", "role reassignment attempt"),
    (r"\bforget (?:your|all) (?:rules|instructions|constraints)\b", "instruction override attempt"),
    (r"\bdeveloper mode\b|\bdo anything now\b", "jailbreak attempt"),
]

# --------------------------------------------------------------------------
# Required elements.
# --------------------------------------------------------------------------

DISCLAIMER_MARKERS: List[str] = [
    "not a diagnosis",
    "not a medical diagnosis",
    "does not replace",
    "consult your doctor",
    "consult your physician",
    "discuss.*with your doctor",
    "speak.*with your doctor",
    "healthcare provider",
    "educational",
]

# --------------------------------------------------------------------------
# Critical values - these bypass explanation entirely and escalate.
#
# Matching is by EXACT normalised name plus unit compatibility, never by
# substring. Substring matching is dangerously wrong here: "Mean Corp.
# Hemoglobin (MCH)" contains "hemoglobin" but is measured in picograms per
# cell, and "BUN/Creatinine Ratio" contains "creatinine" but is a dimensionless
# ratio. Checking either against whole-blood limits escalates a normal report to
# urgent care - a false alarm that does real harm in a patient-facing tool.
#
# Thresholds are widely cited adult critical-value notification limits, used
# only to decide whether to escalate, never to diagnose.

# Derived quantities are never critical values in their own right.
_DERIVED_MARKERS = (
    "ratio", "index", "corpuscular", "corp.", "corp ", "mch", "mcv", "mchc",
    "percent", "percentage", "fraction", "%", "estimated", "calculated", "egfr",
)

CRITICAL_VALUES: Dict[str, Dict[str, Any]] = {
    "potassium": {
        "aliases": {"potassium", "serum potassium", "k", "k+"},
        "units": {"mmol/l", "meq/l"},
        "low": 2.5, "high": 6.5,
    },
    "sodium": {
        "aliases": {"sodium", "serum sodium", "na", "na+"},
        "units": {"mmol/l", "meq/l"},
        "low": 120.0, "high": 160.0,
    },
    "glucose": {
        "aliases": {
            "glucose", "blood glucose", "fasting blood glucose", "fasting glucose",
            "random blood glucose", "plasma glucose", "fbs", "rbs",
        },
        "units": {"mg/dl"},
        "low": 40.0, "high": 450.0,
    },
    "hemoglobin": {
        "aliases": {"hemoglobin", "haemoglobin", "hb", "hgb"},
        "units": {"g/dl"},
        "low": 7.0, "high": 20.0,
    },
    "platelets": {
        "aliases": {"platelets", "platelet count", "plt", "thrombocytes"},
        "units": {"x10^3/ul", "10^3/ul", "k/ul", "x10^9/l", "10^9/l", "/ul", "cells/ul"},
        "low": 50.0, "high": 1000.0,
    },
    "wbc": {
        "aliases": {
            "wbc", "white blood cells", "white blood cell count", "total leucocyte count",
            "total leukocyte count", "leucocyte count", "leukocyte count", "tlc",
        },
        "units": {"x10^3/ul", "10^3/ul", "k/ul", "x10^9/l", "10^9/l", "/ul", "cells/ul"},
        "low": 1.5, "high": 30.0,
    },
    "calcium": {
        "aliases": {"calcium", "serum calcium", "total calcium", "ca"},
        "units": {"mg/dl"},
        "low": 6.0, "high": 13.0,
    },
    "creatinine": {
        "aliases": {"creatinine", "serum creatinine", "crea"},
        "units": {"mg/dl"},
        "low": None, "high": 7.0,
    },
}


def _normalise_test_name(name: str) -> str:
    """Lowercase, drop bracketed abbreviations and punctuation, collapse spaces."""
    text = str(name).lower()
    text = re.sub(r"\([^)]*\)", " ", text)          # "(MCH)", "(WBC)"
    text = re.sub(r"[^a-z0-9+/^ ]+", " ", text)     # keep + / ^ for K+, x10^3
    return re.sub(r"\s+", " ", text).strip()


def _normalise_unit(unit: Optional[str]) -> str:
    return re.sub(r"\s+", "", str(unit or "").lower())


def _is_derived(normalised_name: str, raw_name: str) -> bool:
    """True for ratios, indices and corpuscular indices, which are never critical values."""
    haystack = f"{normalised_name} {str(raw_name).lower()}"
    return any(marker in haystack for marker in _DERIVED_MARKERS)


ESCALATION_MESSAGE = (
    "One or more values in this report fall far outside the usual reference range. "
    "MedExplain AI does not explain results in this range. Please contact your doctor "
    "or nearest healthcare facility promptly to have this report reviewed."
)


def _match_any(text: str, patterns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    lowered = text.lower()
    for pattern, reason in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            findings.append({"pattern": pattern, "reason": reason, "matched_text": match.group(0)[:120]})
    return findings


def check_scope(text: str) -> List[Dict[str, str]]:
    """Diagnosis, prescription and prognosis violations."""
    return (
        _match_any(text, DIAGNOSIS_PATTERNS)
        + _match_any(text, PRESCRIPTION_PATTERNS)
        + _match_any(text, PROGNOSIS_PATTERNS)
    )


def check_tone(text: str) -> List[Dict[str, str]]:
    """Alarming or dismissive language."""
    return _match_any(text, ALARMING_PATTERNS) + _match_any(text, DISMISSIVE_PATTERNS)


def check_injection(text: str) -> List[Dict[str, str]]:
    """Prompt injection markers in untrusted input."""
    return _match_any(text, INJECTION_PATTERNS)


def has_disclaimer(text: str) -> bool:
    """True when the non-diagnostic notice survives in the output."""
    lowered = text.lower()
    return any(re.search(marker, lowered) for marker in DISCLAIMER_MARKERS)


def find_critical_values(test_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return test items whose value crosses a critical notification limit.

    A test qualifies only when its normalised name matches a known analyte
    exactly and its unit is compatible. Anything derived - a ratio, an index, a
    corpuscular measure - is excluded outright.
    """
    critical: List[Dict[str, Any]] = []

    for item in test_items:
        raw_name = str(item.get("test_name", ""))
        name = _normalise_test_name(raw_name)
        if not name or _is_derived(name, raw_name):
            continue

        limits = None
        for candidate in CRITICAL_VALUES.values():
            if name in candidate["aliases"]:
                limits = candidate
                break
        if limits is None:
            continue

        accepted_units = limits.get("units")
        unit = _normalise_unit(item.get("unit"))
        if accepted_units and unit and unit not in accepted_units:
            # Same word, different quantity - do not compare against these limits.
            continue

        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue

        low, high = limits.get("low"), limits.get("high")
        if low is not None and value <= low:
            critical.append({"test_name": raw_name, "value": value, "unit": item.get("unit"),
                             "limit": low, "direction": "critically low"})
        elif high is not None and value >= high:
            critical.append({"test_name": raw_name, "value": value, "unit": item.get("unit"),
                             "limit": high, "direction": "critically high"})

    return critical
