"""
MedExplain AI — NLP Extraction Agent Pattern Definitions
Module: agents.extraction.patterns
"""

import re
from typing import Dict, List, Tuple

# Alias dictionary mapping lab test name variations to canonical names
ALIAS_MAP: Dict[str, str] = {
    # HbA1c
    "hba1c": "hba1c",
    "hemoglobin a1c": "hba1c",
    "glycated hemoglobin": "hba1c",
    "a1c": "hba1c",
    # Fasting Glucose
    "fasting glucose": "fasting_glucose",
    "fbs": "fasting_glucose",
    "fasting blood sugar": "fasting_glucose",
    "glucose fasting": "fasting_glucose",
    "glucose": "fasting_glucose",
    # Haemoglobin
    "haemoglobin": "haemoglobin",
    "hemoglobin": "haemoglobin",
    "hb": "haemoglobin",
    "hgb": "haemoglobin",
    # Cholesterol Total
    "total cholesterol": "total_cholesterol",
    "cholesterol total": "total_cholesterol",
    "cholesterol": "total_cholesterol",
    # LDL Cholesterol
    "ldl": "ldl_cholesterol",
    "ldl cholesterol": "ldl_cholesterol",
    "ldl-c": "ldl_cholesterol",
    "low density lipoprotein": "ldl_cholesterol",
    # HDL Cholesterol
    "hdl": "hdl_cholesterol",
    "hdl cholesterol": "hdl_cholesterol",
    "hdl-c": "hdl_cholesterol",
    "high density lipoprotein": "hdl_cholesterol",
    # Triglycerides
    "triglycerides": "triglycerides",
    "tg": "triglycerides",
    "triglyceride": "triglycerides",
    # Creatinine
    "creatinine": "creatinine",
    "serum creatinine": "creatinine",
    "creat": "creatinine",
    # ALT / SGPT
    "alt": "alt",
    "sgpt": "alt",
    "alanine aminotransferase": "alt",
    # AST / SGOT
    "ast": "ast",
    "sgot": "ast",
    "aspartate aminotransferase": "ast",
    # TSH
    "tsh": "tsh",
    "thyroid stimulating hormone": "tsh",
    "thyrotropin": "tsh",
    # Vitamin D
    "vitamin d": "vitamin_d",
    "25-hydroxy vitamin d": "vitamin_d",
    "vit d": "vitamin_d",
    "25-oh vit d": "vitamin_d",
    # White Blood Cell Count
    "wbc": "white_cell_count",
    "white cell count": "white_cell_count",
    "white blood cell count": "white_cell_count",
    "leukocytes": "white_cell_count",
    "total wbc": "white_cell_count",
}

# General regex pattern for extracting test name, value, unit, reference range, and flag
# Example match: "Hemoglobin: 14.2 g/dL (13.5 - 17.5) [NORMAL]"
# Example match: "HbA1c  8.2 %  4.0-5.6 HIGH"
LAB_LINE_REGEX = re.compile(
    r"(?P<test_name>[A-Za-z0-9\s\-\/\(\)\.\_]{2,35})"  # Test name
    r"[:\=\-\s]+"                                      # Separator
    r"(?P<value>\d+(?:\.\d+)?)"                         # Numeric value
    r"\s*"                                             # Optional space
    r"(?P<unit>[a-zA-Z\%\^\d\/\.\,\*]+)?"              # Unit
    r"\s*"                                             # Optional space
    r"(?:[\(\[\{]?(?P<ref_range>\d+(?:\.\d+)?\s*[\-\–\—\:]\s*\d+(?:\.\d+)?|[\<\>]\s*\d+(?:\.\d+)?)[\]\}\)]?)?" # Reference range
    r"\s*"
    r"(?P<flag>HIGH|LOW|NORMAL|ABNORMAL|H|L|N)?"        # Flag
    , re.IGNORECASE
)

# Standard reference ranges fallback dictionary if missing from line
DEFAULT_REF_RANGES: Dict[str, Tuple[float, float]] = {
    "hba1c": (4.0, 5.6),             # %
    "fasting_glucose": (70.0, 99.0), # mg/dL
    "haemoglobin": (13.5, 17.5),     # g/dL
    "total_cholesterol": (125.0, 199.0), # mg/dL
    "ldl_cholesterol": (0.0, 99.0),  # mg/dL
    "hdl_cholesterol": (40.0, 60.0), # mg/dL
    "triglycerides": (0.0, 149.0),   # mg/dL
    "creatinine": (0.74, 1.35),      # mg/dL
    "alt": (7.0, 56.0),              # U/L
    "ast": (10.0, 40.0),             # U/L
    "tsh": (0.4, 4.0),               # mIU/L
    "vitamin_d": (30.0, 100.0),      # ng/mL
    "white_cell_count": (4.5, 11.0), # x10^3/uL
}
