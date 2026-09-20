"""
MedExplain AI - Centralised logging with per-run trace IDs.
Module: medexplain.logging_config

Every pipeline run is assigned a trace_id. Each agent hop, MCP tool call and
safety verdict is logged against that trace_id, producing the audit trail that
the Responsible AI accountability control and the individual security
assessments both depend on.

Per AGENTS.md: no print() for logging.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="-")

LOG_DIR = Path(os.getenv("MEDEXPLAIN_LOG_DIR", "logs"))
AUDIT_LOG = LOG_DIR / "audit.jsonl"


class TraceIdFilter(logging.Filter):
    """Injects the current trace_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _TRACE_ID.get()
        return True


def new_trace_id() -> str:
    """Start a new trace and return its id."""
    trace_id = uuid.uuid4().hex[:12]
    _TRACE_ID.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """Return the trace id for the current run."""
    return _TRACE_ID.get()


def set_trace_id(trace_id: str) -> None:
    """Adopt an externally supplied trace id (e.g. from a request header)."""
    _TRACE_ID.set(trace_id)


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once, for console output."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if getattr(root, "_medexplain_configured", False):
        return

    # stderr, not stdout: the MCP server speaks JSON-RPC over stdout, so a log
    # line written there corrupts the protocol stream.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(trace_id)s] [%(name)s] [%(levelname)s] %(message)s")
    )
    handler.addFilter(TraceIdFilter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level_name)
    root._medexplain_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    configure_logging()
    return logging.getLogger(name)


def audit(event: str, **fields: Any) -> Dict[str, Any]:
    """
    Append a structured audit record to logs/audit.jsonl.

    This is the evidence trail the individual security assessments screenshot:
    agent hops, MCP tool calls, retrieved source ids and safety verdicts.
    """
    record: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": get_trace_id(),
        "event": event,
    }
    record.update(fields)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:  # auditing must never break the pipeline
        get_logger("MedExplain.Audit").warning("Could not write audit record for event %s", event)

    return record
