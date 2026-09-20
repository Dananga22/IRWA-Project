"""
MedExplain AI - LangGraph orchestrator.
Module: medexplain.graph

Five agents as nodes on a state machine. Agents never call each other; they read
and write the shared MedicalReportState, and the graph decides what runs next.

    document -> extraction -> retrieval -> explanation -> safety
                                               ^             |
                                               |             | approved == False
                                               +-------------+  (max 2 revisions)

The conditional edge from safety back to explanation is what makes this a graph
rather than a chain: the Safety Verification Agent can reject a draft and send
it back for revision, and the system fails closed once retries are exhausted.

If langgraph is not installed the same graph is executed by a small built-in
runner with identical semantics, so the pipeline always runs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from medexplain.logging_config import audit, get_logger, new_trace_id
from medexplain.state import MedicalReportState, new_state
from medexplain.security import sanitize_document_text

logger = get_logger("MedExplain.Graph")

MAX_REVISIONS = int(os.getenv("MAX_SAFETY_REVISIONS", "2"))

try:
    from langgraph.graph import END, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    HAS_LANGGRAPH = False


# ---------------------------------------------------------------- node: agent 1

def document_processing_node(state: MedicalReportState) -> Dict[str, Any]:
    """[AGENT 1] Read the PDF and write raw_text to the shared state."""
    import pymupdf

    pdf_path = state["pdf_path"]
    logger.info("[AGENT 1: Document Processing] reading %s", pdf_path)

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.error("Could not open PDF %s: %s", pdf_path, exc)
        audit("agent.document", outcome="error", error=str(exc))
        return {"status": "failed", "errors": state.get("errors", []) + [f"document: {exc}"]}

    pages = [page.get_text() for page in document]
    raw_text = "\n".join(pages)
    document.close()

    # Text inside the report is untrusted input, not instructions.
    sanitised = sanitize_document_text(raw_text)

    audit(
        "agent.document",
        outcome="ok",
        pages=len(pages),
        characters=len(raw_text),
        injection_findings=len(sanitised["injection_findings"]),
    )
    logger.info("[AGENT 1] extracted %d characters from %d page(s)", len(raw_text), len(pages))

    return {"raw_text": str(sanitised["text"]), "page_count": len(pages), "status": "document_processed"}


# ---------------------------------------------------------------- node: agent 2

def extraction_node(state: MedicalReportState) -> Dict[str, Any]:
    """
    [AGENT 2] Two NLP tasks: entity extraction and findings summarisation.

    The brief names NER and Summarization as the NLP techniques; this node does
    the entity side (test names, values, units, ranges) and the deterministic
    summary of what the report contains.
    """
    from medexplain.extraction import extract_metadata, extract_test_items
    from medexplain.summarisation import summarise_findings

    logger.info("[AGENT 2: NLP Extraction] parsing report content")

    raw_text = state.get("raw_text", "")
    metadata = extract_metadata(raw_text)
    test_items = extract_test_items(state["pdf_path"], raw_text)
    abnormal = [i for i in test_items if str(i.get("flag", "")).upper() in {"HIGH", "LOW"}]

    findings_summary = summarise_findings(metadata, test_items)

    audit("agent.extraction", outcome="ok", items=len(test_items), abnormal=len(abnormal))
    logger.info("[AGENT 2] extracted %d test item(s), %d outside range", len(test_items), len(abnormal))
    logger.info("[AGENT 2] findings summary: %s", findings_summary["text"])

    return {
        "metadata": metadata,
        "test_items": test_items,
        "total_tests": len(test_items),
        "abnormal_count": len(abnormal),
        "findings_summary": findings_summary,
        "status": "extracted",
    }


# ---------------------------------------------------------------- node: agent 3

def retrieval_node(state: MedicalReportState) -> Dict[str, Any]:
    """
    [AGENT 3] Retrieve grounding context through the MCP retrieval tool, then
    summarise what came back.

    Query-biased extractive summarisation of retrieved documents is a standard
    information-retrieval task - it is what produces the snippet under a search
    result. Here it yields the cited key points shown above the explanation.
    """
    from agents.retrieval.mcp_client import get_mcp_client
    from medexplain.summarisation import textrank_summary

    logger.info("[AGENT 3: RAG Retrieval] issuing MCP tool calls")

    client = get_mcp_client()
    context: Dict[str, Any] = {}
    all_passages: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    seen_passage_ids: set = set()
    backend = ""
    transport = client.transport

    # One MCP session for the whole report rather than one per analyte.
    requests = [
        {"analyte": str(item.get("test_name", "")).strip(), "flag": item.get("flag"), "top_k": 3}
        for item in state.get("test_items", [])
        if str(item.get("test_name", "")).strip()
    ]
    results = client.call_tools(requests)

    for request, result in zip(requests, results):
        name = request["analyte"]
        context[name] = result
        backend = result.get("backend", backend)
        transport = result.get("transport", transport)

        for passage in result.get("passages", []):
            if passage["id"] not in seen_passage_ids:
                seen_passage_ids.add(passage["id"])
                all_passages.append(passage)
        for citation in result.get("citations", []):
            if citation not in citations:
                citations.append(citation)

    grounded_count = sum(1 for c in context.values() if c.get("grounded"))
    audit(
        "agent.retrieval.summary",
        analytes=len(context),
        grounded=grounded_count,
        unique_passages=len(all_passages),
        backend=backend,
        transport=transport,
    )
    logger.info(
        "[AGENT 3] grounded %d/%d analyte(s) using %d unique passage(s) | backend=%s transport=%s",
        grounded_count,
        len(context),
        len(all_passages),
        backend,
        transport,
    )

    key_points = textrank_summary(
        all_passages,
        top_n=4,
        query_terms=[i.get("test_name", "") for i in state.get("test_items", [])],
    )
    audit(
        "agent.summarisation",
        method="textrank+tfidf",
        passages=len(all_passages),
        key_points=len(key_points),
    )
    logger.info("[AGENT 3] TextRank produced %d key point(s) from the retrieved sources", len(key_points))

    return {
        "retrieved_context": context,
        "retrieved_passages": all_passages,
        "citations": citations,
        "retrieval_backend": backend,
        "mcp_transport": transport,
        "key_points": key_points,
        "summarisation_method": "TextRank (TF-IDF sentence vectors + PageRank centrality)",
        "status": "retrieved",
    }


# ---------------------------------------------------------------- node: agent 4

def explanation_node(state: MedicalReportState) -> Dict[str, Any]:
    """[AGENT 4] Write the grounded explanation (or the revision)."""
    from agents.explanation.grounded_explainer import get_explanation_agent

    revision = state.get("revision_count", 0)
    hint = state.get("revision_hint")
    label = f"revision {revision}" if revision else "first draft"
    logger.info("[AGENT 4: LLM Explanation] generating %s", label)

    agent = get_explanation_agent()
    explanation = agent.explain(
        test_items=state.get("test_items", []),
        retrieved_context=state.get("retrieved_context", {}),
        revision_hint=hint,
        findings_summary=state.get("findings_summary", {}),
    )

    return {
        "explanation": explanation,
        "explanation_model": agent.model_label,
        "status": "explained",
    }


# ---------------------------------------------------------------- node: agent 5

def safety_node(state: MedicalReportState) -> Dict[str, Any]:
    """[AGENT 5] Review the draft and set the approved flag."""
    from agents.safety.verifier import get_safety_agent

    logger.info("[AGENT 5: Safety Verification] reviewing draft")

    verdict = get_safety_agent().verify(
        explanation=state.get("explanation", ""),
        retrieved_passages=state.get("retrieved_passages", []),
        test_items=state.get("test_items", []),
        citations=state.get("citations", []),
    )

    update: Dict[str, Any] = {
        "approved": verdict.approved,
        "safety_verdict": verdict.to_dict(),
        "revision_hint": verdict.revision_hint,
        "escalation_message": verdict.escalation_message,
    }

    if verdict.critical_values:
        update["explanation"] = verdict.escalation_message or ""
        update["status"] = "escalated"
    elif verdict.approved:
        update["status"] = "approved"
    else:
        update["status"] = "rejected"
        update["revision_count"] = state.get("revision_count", 0) + 1

    return update


# ------------------------------------------------------------- conditional edge

def route_after_safety(state: MedicalReportState) -> str:
    """
    The feedback edge.

    approved            -> finish
    critical values     -> finish (escalation replaces the explanation)
    rejected, retries left -> back to the explanation agent
    rejected, exhausted -> finish, failing closed
    """
    if state.get("approved"):
        return "finish"

    verdict = state.get("safety_verdict", {})
    if verdict.get("critical_values"):
        logger.warning("[GRAPH] critical values present - escalating, no revision attempted")
        return "finish"

    revisions = state.get("revision_count", 0)
    if revisions <= MAX_REVISIONS:
        logger.warning("[GRAPH] safety rejected the draft - routing back to explanation (attempt %d/%d)", revisions, MAX_REVISIONS)
        audit("graph.revision", attempt=revisions, max_revisions=MAX_REVISIONS)
        return "revise"

    logger.error("[GRAPH] safety rejected after %d revision(s) - failing closed", revisions - 1)
    audit("graph.fail_closed", revisions=revisions - 1)
    return "finish"


def blocked_node(state: MedicalReportState) -> Dict[str, Any]:
    """Terminal node when the pipeline fails closed after exhausted revisions."""
    if state.get("approved") or state.get("safety_verdict", {}).get("critical_values"):
        return {}
    return {
        "status": "blocked",
        "explanation": (
            "MedExplain AI could not produce an explanation for this report that passed its own "
            "safety review, so nothing is being shown. Please ask your doctor to review the report."
        ),
    }


# ---------------------------------------------------------------------- builder

def build_graph():
    """Compile the LangGraph state machine."""
    if not HAS_LANGGRAPH:
        return None

    graph = StateGraph(MedicalReportState)
    graph.add_node("document", document_processing_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("safety", safety_node)
    graph.add_node("finalise", blocked_node)

    graph.set_entry_point("document")
    graph.add_edge("document", "extraction")
    graph.add_edge("extraction", "retrieval")
    graph.add_edge("retrieval", "explanation")
    graph.add_edge("explanation", "safety")
    graph.add_conditional_edges(
        "safety",
        route_after_safety,
        {"revise": "explanation", "finish": "finalise"},
    )
    graph.add_edge("finalise", END)

    return graph.compile()


def _run_sequential(state: MedicalReportState) -> MedicalReportState:
    """Fallback runner with the same nodes, edges and retry semantics."""
    nodes: List[Callable[[MedicalReportState], Dict[str, Any]]] = [
        document_processing_node,
        extraction_node,
        retrieval_node,
    ]
    for node in nodes:
        state.update(node(state))  # type: ignore[arg-type]
        if state.get("status") == "failed":
            return state

    while True:
        state.update(explanation_node(state))  # type: ignore[arg-type]
        state.update(safety_node(state))  # type: ignore[arg-type]
        decision = route_after_safety(state)
        if decision == "finish":
            state.update(blocked_node(state))  # type: ignore[arg-type]
            return state


_COMPILED = None


def run_pipeline(pdf_path: str, trace_id: Optional[str] = None) -> MedicalReportState:
    """
    Execute the full five-agent pipeline for one report.

    Returns the final shared state, including the safety verdict and citations.
    """
    global _COMPILED

    trace = trace_id or new_trace_id()
    state = new_state(pdf_path=str(Path(pdf_path)), trace_id=trace)

    audit("pipeline.start", pdf_path=str(pdf_path), orchestrator="langgraph" if HAS_LANGGRAPH else "sequential-fallback")
    logger.info("Pipeline start | orchestrator=%s | file=%s", "langgraph" if HAS_LANGGRAPH else "sequential-fallback", pdf_path)

    if HAS_LANGGRAPH:
        if _COMPILED is None:
            _COMPILED = build_graph()
        # recursion_limit guards the revise loop at the graph level too
        final = _COMPILED.invoke(state, config={"recursion_limit": 25})
        result: MedicalReportState = dict(final)  # type: ignore[assignment]
    else:
        logger.warning("langgraph not installed - running the equivalent sequential graph runner")
        result = _run_sequential(state)

    audit(
        "pipeline.end",
        status=result.get("status"),
        approved=result.get("approved"),
        revisions=result.get("revision_count", 0),
        tests=result.get("total_tests", 0),
        citations=len(result.get("citations", [])),
    )
    logger.info(
        "Pipeline end | status=%s | approved=%s | revisions=%d",
        result.get("status"),
        result.get("approved"),
        result.get("revision_count", 0),
    )
    return result


def run_preview_pipeline(test_items: List[Dict[str, Any]], trace_id: Optional[str] = None) -> MedicalReportState:
    """
    Execute the partial AI pipeline (Retrieval -> Explanation -> Safety) for previewing
    or generating summaries directly from structured test items without a PDF.
    """
    trace = trace_id or new_trace_id()
    # Initialize state with test items, skipping document and extraction steps.
    state = new_state(pdf_path="manual-entry-preview", trace_id=trace)
    state.update({
        "test_items": test_items,
        "total_tests": len(test_items),
        "abnormal_count": len([i for i in test_items if str(i.get("flag", "")).upper() in {"HIGH", "LOW"}]),
        "status": "extracted",
        # Mock empty summary/metadata since we skipped extraction of raw doc text
        "metadata": {},
        "findings_summary": {"text": "Manual input provided by lab assistant."},
    })

    audit("pipeline.start_preview", items=len(test_items))
    logger.info("Pipeline preview start | items=%d", len(test_items))

    # We use a sequential loop for the preview to avoid duplicating graph edges 
    # and ensuring we can easily run without langgraph graph compilation dependencies.
    nodes: List[Callable[[MedicalReportState], Dict[str, Any]]] = [retrieval_node]
    for node in nodes:
        state.update(node(state))  # type: ignore[arg-type]
        if state.get("status") == "failed":
            return state

    while True:
        state.update(explanation_node(state))  # type: ignore[arg-type]
        state.update(safety_node(state))  # type: ignore[arg-type]
        decision = route_after_safety(state)
        if decision == "finish":
            state.update(blocked_node(state))  # type: ignore[arg-type]
            break

    audit(
        "pipeline.preview_end",
        status=state.get("status"),
        approved=state.get("approved"),
        revisions=state.get("revision_count", 0),
    )
    return state

