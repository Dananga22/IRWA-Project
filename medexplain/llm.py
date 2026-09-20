"""
MedExplain AI - Shared Gemini client construction and diagnosis.
Module: medexplain.llm

Both the Explanation Agent and the Localisation Agent need a Gemini client, and
both must degrade gracefully without one. Building the client in one place means
they degrade for the same reasons and, more importantly, report the *same*
reason - "no API key" and "SDK not installed" are different problems with
different fixes, and conflating them wastes debugging time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

from medexplain.logging_config import get_logger

logger = get_logger("MedExplain.LLM")

PLACEHOLDER_KEYS = {"your_gemini_api_key_here", "your-gemini-api-key-here", ""}


def find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """Locate the .env file the way python-dotenv does, for diagnostics."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def diagnose() -> dict:
    """
    Explain the current LLM configuration state.

    Returned by /api/v1/health and printed by scripts/test_gemini.py, so a
    missing key is visible without reading server logs.
    """
    dotenv_path = find_dotenv()
    raw_key = os.getenv("GEMINI_API_KEY", "")
    key_set = bool(raw_key.strip()) and raw_key.strip() not in PLACEHOLDER_KEYS

    try:
        from google import genai  # noqa: F401

        sdk_installed = True
    except ImportError:
        sdk_installed = False

    if not sdk_installed:
        reason = "google-genai is not installed - run: pip install -U google-genai"
    elif not raw_key.strip():
        reason = (
            "GEMINI_API_KEY is not set. "
            + (f"A .env file was found at {dotenv_path}, but it has no usable GEMINI_API_KEY line."
               if dotenv_path else
               "No .env file was found - copy .env.example to .env and add your key.")
        )
    elif not key_set:
        reason = "GEMINI_API_KEY is still the placeholder value from .env.example."
    else:
        reason = "configured"

    return {
        "sdk_installed": sdk_installed,
        "key_set": key_set,
        "key_length": len(raw_key.strip()),
        "dotenv_found": str(dotenv_path) if dotenv_path else None,
        "model": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "reason": reason,
    }


def build_client(api_key: Optional[str] = None, purpose: str = "LLM") -> Tuple[Optional[Any], str]:
    """
    Return (client, reason).

    `client` is None when a live model is unavailable; `reason` always says why,
    in words that name the fix.
    """
    try:
        from google import genai
    except ImportError:
        reason = "google-genai is not installed - run: pip install -U google-genai"
        logger.warning("%s falling back: %s", purpose, reason)
        return None, reason

    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if key in PLACEHOLDER_KEYS:
        state = diagnose()
        logger.warning("%s falling back: %s", purpose, state["reason"])
        return None, state["reason"]

    try:
        client = genai.Client(api_key=key)
    except Exception as exc:
        reason = f"Gemini client could not be created: {exc}"
        logger.warning("%s falling back: %s", purpose, reason)
        return None, reason

    return client, "configured"
