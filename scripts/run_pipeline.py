"""
MedExplain AI - End-to-end pipeline runner with a visible agent trace.
Script: scripts/run_pipeline.py

This is the script to project during the live demonstration. It prints the hop
between every agent, the MCP tool calls, the retrieved sources and the safety
verdict, so the multi-agent behaviour is visible rather than asserted.

Usage:
    python scripts/run_pipeline.py sample_reports/sample_cbc_report.pdf
    python scripts/run_pipeline.py sample_reports/sample_lipid_panel.pdf --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medexplain.graph import HAS_LANGGRAPH, run_pipeline  # noqa: E402
from medexplain.logging_config import AUDIT_LOG, configure_logging  # noqa: E402

RULE = "=" * 78


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MedExplain AI five-agent pipeline.")
    parser.add_argument("pdf", nargs="?", default="sample_reports/sample_cbc_report.pdf", help="Path to a lab report PDF")
    parser.add_argument("--json", action="store_true", help="Print the final state as JSON instead of a report")
    args = parser.parse_args()

    configure_logging()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 1

    print(RULE)
    print(" MedExplain AI - five-agent pipeline")
    print(f" Orchestrator : {'LangGraph state machine' if HAS_LANGGRAPH else 'sequential fallback runner'}")
    print(f" Report       : {pdf_path}")
    print(RULE)

    state = run_pipeline(str(pdf_path))

    if args.json:
        print(json.dumps(state, indent=2, default=str))
        return 0

    verdict = state.get("safety_verdict", {})

    print()
    print(RULE)
    print(" RUN SUMMARY")
    print(RULE)
    print(f" trace id            : {state.get('trace_id')}")
    print(f" status              : {state.get('status')}")
    print(f" tests extracted     : {state.get('total_tests')} ({state.get('abnormal_count')} outside range)")
    print(f" retrieval backend   : {state.get('retrieval_backend')}")
    print(f" MCP transport       : {state.get('mcp_transport')}")
    print(f" explanation model   : {state.get('explanation_model')}")
    print(f" safety approved     : {state.get('approved')}")
    print(f" groundedness        : {verdict.get('groundedness_score', 0):.0%}")
    print(f" safety revisions    : {state.get('revision_count', 0)}")
    print(f" audit log           : {AUDIT_LOG}")

    checks = verdict.get("checks", {})
    if checks:
        print()
        print(" SAFETY CHECKS")
        for name, passed in checks.items():
            print(f"   [{'PASS' if passed else 'FAIL'}] {name}")

    violations = verdict.get("violations", [])
    if violations:
        print()
        print(" SAFETY VIOLATIONS")
        for violation in violations:
            print(f"   - {violation['check']}: {violation['reason']}")

    findings = state.get("findings_summary", {})
    if findings.get("text"):
        print()
        print(" FINDINGS SUMMARY (Agent 2, computed from the values)")
        print(f"   {findings['text']}")

    key_points = state.get("key_points", [])
    if key_points:
        print()
        print(f" KEY POINTS ({state.get('summarisation_method', 'summarisation')})")
        for point in key_points:
            print(f"   - {point['sentence']}")
            print(f"     [{point['source']}] {point['title']}")

    citations = state.get("citations", [])
    if citations:
        print()
        print(" SOURCES CITED")
        seen = set()
        for citation in citations:
            key = citation["source_url"]
            if key in seen:
                continue
            seen.add(key)
            print(f"   - [{citation['source']}] {citation['title']}")
            print(f"     {citation['source_url']}")

    print()
    print(RULE)
    print(" EXPLANATION DELIVERED TO PATIENT")
    print(RULE)
    print(state.get("explanation", "(none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
