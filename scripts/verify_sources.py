"""
MedExplain AI - Verify every knowledge base source URL still resolves.
Script: scripts/verify_sources.py

Run this before an evaluation. A citation pointing at a dead link is worse than
no citation, and source reliability is directly assessed in the Information
Retrieval security specialisation.

Usage:
    python scripts/verify_sources.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT_ROOT / "knowledge_base" / "corpus.json"

TIMEOUT_SECONDS = 15
USER_AGENT = "MedExplain-AI-SourceVerifier/1.0 (university project)"


def check(url: str) -> Tuple[bool, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 400, str(response.status)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, type(exc).__name__


def main() -> int:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    passages: List[Dict] = data["passages"]

    urls = sorted({p["source_url"] for p in passages})
    print(f"Verifying {len(urls)} unique source URL(s) across {len(passages)} passages\n")

    failures: List[Tuple[str, str]] = []
    for url in urls:
        ok, detail = check(url)
        print(f"  [{'OK ' if ok else 'FAIL'}] {detail:<12} {url}")
        if not ok:
            failures.append((url, detail))

    print()
    if failures:
        print(f"{len(failures)} source(s) could not be verified:")
        for url, detail in failures:
            print(f"  - {url} ({detail})")
        print("\nUpdate or replace these entries in knowledge_base/corpus.json before presenting.")
        return 1

    print("All source URLs resolved successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
