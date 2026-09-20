"""
MedExplain AI - Shared state object passed between agents.
Module: medexplain.state

This is the agent communication contract inside the pipeline. Agents are nodes
on a LangGraph state machine; they communicate by reading and writing keys on
this typed object rather than by calling each other directly.

Ownership - each key is written by exactly one agent:

    raw_text            -> Agent 1  Document Processing
    metadata            -> Agent 2  NLP Extraction
    test_items          -> Agent 2  NLP Extraction
    findings_summary    -> Agent 2  NLP Extraction (summarisation)
    retrieved_context   -> Agent 3  RAG Retrieval (via MCP tool call)
    citations           -> Agent 3  RAG Retrieval
    key_points          -> Agent 3  RAG Retrieval (query-biased summarisation)
    explanation         -> Agent 4  LLM Explanation
    approved            -> Agent 5  Safety Verification
    safety_verdict      -> Agent 5  Safety Verification
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.12+ / typing_extensions
    from typing import TypedDict, NotRequired
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict, NotRequired  # type: ignore


class MedicalReportState(TypedDict, total=False):
    """Typed shared state for one report run."""

    # run identity
    trace_id: str
    pdf_path: str

    # Agent 1 - Document Processing
    raw_text: str
    page_count: int

    # Agent 2 - NLP Extraction (entity extraction + findings summarisation)
    metadata: Dict[str, Any]
    test_items: List[Dict[str, Any]]
    total_tests: int
    abnormal_count: int
    findings_summary: Dict[str, Any]

    # Agent 3 - RAG Retrieval (written via MCP tool call)
    retrieved_context: Dict[str, Any]
    retrieved_passages: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    retrieval_backend: str
    mcp_transport: str
    key_points: List[Dict[str, Any]]
    summarisation_method: str

    # Agent 4 - LLM Explanation
    explanation: str
    explanation_model: str
    revision_count: int
    revision_hint: NotRequired[Optional[str]]

    # Agent 5 - Safety Verification
    approved: bool
    safety_verdict: Dict[str, Any]
    escalation_message: NotRequired[Optional[str]]

    # terminal
    status: str
    errors: List[str]


def new_state(pdf_path: str, trace_id: str) -> MedicalReportState:
    """Create an empty state for a run."""
    return MedicalReportState(
        trace_id=trace_id,
        pdf_path=pdf_path,
        raw_text="",
        page_count=0,
        metadata={},
        test_items=[],
        total_tests=0,
        abnormal_count=0,
        findings_summary={},
        retrieved_context={},
        retrieved_passages=[],
        citations=[],
        retrieval_backend="",
        mcp_transport="",
        key_points=[],
        summarisation_method="",
        explanation="",
        explanation_model="",
        revision_count=0,
        revision_hint=None,
        approved=False,
        safety_verdict={},
        escalation_message=None,
        status="initialised",
        errors=[],
    )
