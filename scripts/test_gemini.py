"""
MedExplain AI - Verify the Gemini API key, model and thinking level.
Script: scripts/test_gemini.py

Run this first after adding a key to .env. It checks four things in order and
tells you exactly which one failed, so a broken key never surfaces for the first
time in the middle of a live demonstration.

Usage:
    python scripts/test_gemini.py
    python scripts/test_gemini.py --model gemini-3.5-flash-lite
    python scripts/test_gemini.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from medexplain.llm import diagnose, find_dotenv  # noqa: E402

RULE = "=" * 74


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Gemini API key and model.")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    parser.add_argument("--list", action="store_true", help="List every model this key can use")
    args = parser.parse_args()

    print(RULE)
    print(" MedExplain AI - Gemini connectivity check")
    print(RULE)

    # 0. Where is configuration being read from?
    print(f" cwd                : {Path.cwd()}")
    dotenv_path = find_dotenv()
    if dotenv_path:
        print(f" [ OK ] .env found  : {dotenv_path}")
        try:
            lines = dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            lines = []
            print(f" [WARN] could not read .env: {exc}")
        key_lines = [ln for ln in lines if ln.strip().startswith("GEMINI_API_KEY")]
        if not key_lines:
            print(" [FAIL] .env has no GEMINI_API_KEY line.")
        else:
            raw = key_lines[0].split("=", 1)[-1]
            if raw != raw.strip():
                print(" [WARN] the GEMINI_API_KEY line has surrounding whitespace.")
            if raw.strip().startswith(("\'", '"')):
                print(" [WARN] the key is wrapped in quotes - remove them, .env needs no quoting.")
    else:
        print(" [FAIL] No .env file found from this directory upwards.")
        print("        Fix: copy .env.example .env   (run from the project root)")
    print()

    # 1. SDK present?
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print(" [FAIL] google-genai is not installed.")
        print("        Fix: pip install -U google-genai")
        return 1
    print(" [ OK ] google-genai SDK is installed")

    # 2. Key present?
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        print(" [FAIL] GEMINI_API_KEY is not set in .env")
        print("        The system still runs - it will use the grounded fallback generator.")
        return 1
    print(f" [ OK ] GEMINI_API_KEY is set ({api_key[:6]}...{api_key[-4:]}, {len(api_key)} chars)")

    # 3. Key accepted?
    try:
        client = genai.Client(api_key=api_key)
        available = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception as exc:
        print(f" [FAIL] The API rejected this key: {type(exc).__name__}: {str(exc)[:200]}")
        print("        Get a key at https://aistudio.google.com/app/apikey")
        return 1
    print(f" [ OK ] Key accepted - {len(available)} model(s) available")

    if args.list:
        print("\n All models available to this key:")
        for name in sorted(available):
            print(f"   - {name}")
        print()

    # 4. Requested model usable?
    if args.model not in available:
        print(f" [WARN] {args.model!r} is not in this key's model list.")
        suggestions = [m for m in available if "flash" in m][:8]
        if suggestions:
            print("        Flash models this key can use:")
            for name in sorted(suggestions):
                print(f"          - {name}")
        print(f"        Set GEMINI_MODEL in .env to one of these.")
    else:
        print(f" [ OK ] {args.model} is available to this key")

    # 5. Real generation, with the same config the explanation agent uses.
    is_gemini_3 = args.model.lower().startswith("gemini-3")
    thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "LOW").upper()

    config_kwargs = {"system_instruction": "You are a test harness. Answer in exactly one short sentence."}
    if is_gemini_3:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            print(f" [ OK ] thinking_level={thinking_level} accepted by this SDK version")
        except Exception as exc:
            print(f" [WARN] This SDK cannot set thinking_level ({exc}). Model default will apply.")
            print("        Fix: pip install -U google-genai")
    else:
        config_kwargs["temperature"] = 0.2

    try:
        response = client.models.generate_content(
            model=args.model,
            contents="Reply with exactly: MedExplain connectivity check passed.",
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = (response.text or "").strip()
    except Exception as exc:
        print(f" [FAIL] Generation failed: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    print(f" [ OK ] Generation succeeded")
    print(f"        model replied: {text[:120]!r}")

    usage = getattr(response, "usage_metadata", None)
    if usage:
        thoughts = getattr(usage, "thoughts_token_count", None) or 0
        print(
            f"        tokens: {getattr(usage, 'prompt_token_count', '?')} in, "
            f"{getattr(usage, 'candidates_token_count', '?')} out"
            + (f", {thoughts} thinking" if thoughts else "")
        )

    print()
    state = diagnose()
    print(f"        the app sees: {state['reason']}")
    print()
    print(RULE)
    print(" All checks passed. The explanation and localisation agents will use the live model.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
