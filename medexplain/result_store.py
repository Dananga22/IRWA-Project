"""
MedExplain AI - Durable store for approved pipeline results.
Module: medexplain.result_store

The localisation endpoint translates an explanation looked up by trace id, never
one supplied by the caller - accepting text from a client would let anyone post
arbitrary content and have the model render it, routing around the safety gate.

That lookup was originally an in-process dictionary, which had an obvious flaw:
any server restart emptied it, and a page still showing valid results would fail
to translate with "no approved result found". Development servers restart on
every file save, so this happened constantly.

Results are therefore written to disk. Two data-protection constraints shape
what is stored:

  * Only the approved explanation and the measured values are kept. Patient
    metadata (name, report date) is deliberately not written.
  * Entries expire. Anything older than RESULT_RETENTION_HOURS is pruned on
    access, and the newest RESULT_STORE_LIMIT entries are kept.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from medexplain.logging_config import get_logger

logger = get_logger("MedExplain.ResultStore")

STORE_DIR = Path(os.getenv("MEDEXPLAIN_LOG_DIR", "logs")) / "results"
RETENTION_SECONDS = float(os.getenv("RESULT_RETENTION_HOURS", "24")) * 3600
STORE_LIMIT = int(os.getenv("RESULT_STORE_LIMIT", "200"))

# Trace ids are generated as 12 hex characters. Validating the shape stops a
# crafted id from being used to read an arbitrary path.
_TRACE_RE = re.compile(r"^[0-9a-f]{6,64}$")

_MEMORY: Dict[str, Dict[str, Any]] = {}


def _path_for(trace_id: str) -> Optional[Path]:
    if not _TRACE_RE.match(trace_id or ""):
        logger.warning("Rejected malformed trace id: %r", trace_id)
        return None
    return STORE_DIR / f"{trace_id}.json"


def save(trace_id: str, state: Dict[str, Any]) -> bool:
    """Persist an approved result. Returns True when it was stored."""
    if not trace_id or not state.get("approved"):
        return False

    path = _path_for(trace_id)
    if path is None:
        return False

    payload = {
        "trace_id": trace_id,
        "saved_at": time.time(),
        "explanation": state.get("explanation", ""),
        # Values only - no patient metadata is written to disk.
        "test_items": [
            {
                "test_name": item.get("test_name"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "reference_range": item.get("reference_range"),
                "flag": item.get("flag"),
            }
            for item in state.get("test_items", [])
        ],
    }

    _MEMORY[trace_id] = payload
    try:
        STORE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        # In-memory copy still serves this process, so this is not fatal.
        logger.warning("Could not persist result %s: %s", trace_id, exc)
        return True

    prune()
    return True


def load(trace_id: str) -> Optional[Dict[str, Any]]:
    """Return a stored result, or None if it is unknown or expired."""
    cached = _MEMORY.get(trace_id)
    if cached is not None:
        return cached

    path = _path_for(trace_id)
    if path is None or not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read stored result %s: %s", trace_id, exc)
        return None

    if time.time() - float(payload.get("saved_at", 0)) > RETENTION_SECONDS:
        logger.info("Stored result %s has expired", trace_id)
        path.unlink(missing_ok=True)
        return None

    _MEMORY[trace_id] = payload
    return payload


def prune() -> int:
    """Delete expired entries, then trim to the newest STORE_LIMIT. Returns count removed."""
    if not STORE_DIR.is_dir():
        return 0

    removed = 0
    now = time.time()
    entries: List[tuple] = []

    for path in STORE_DIR.glob("*.json"):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age > RETENTION_SECONDS:
            path.unlink(missing_ok=True)
            _MEMORY.pop(path.stem, None)
            removed += 1
        else:
            entries.append((path.stat().st_mtime, path))

    entries.sort(reverse=True)
    for _, path in entries[STORE_LIMIT:]:
        path.unlink(missing_ok=True)
        _MEMORY.pop(path.stem, None)
        removed += 1

    if removed:
        logger.info("Pruned %d stored result(s)", removed)
    return removed


def count() -> int:
    """Number of results currently retrievable."""
    if not STORE_DIR.is_dir():
        return len(_MEMORY)
    return len(list(STORE_DIR.glob("*.json")))
