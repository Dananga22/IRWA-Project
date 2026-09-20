"""
MedExplain AI - MCP server exposing the RAG Retrieval Agent as a tool.
Module: agents.retrieval.mcp_server

This is the agent communication protocol boundary required by the assignment.
The RAG Retrieval Agent is published as a Model Context Protocol tool so that
the LLM Explanation Agent calls it on demand - deciding how many lookups a
report needs - rather than receiving a fixed hand-off.

Run standalone (stdio transport):

    python -m agents.retrieval.mcp_server

Inspect the tool list without writing a client:

    python -m agents.retrieval.mcp_server --list-tools
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

# The MCP Python SDK renamed its ergonomic server class: FastMCP (1.x) became
# MCPServer under mcp.server.mcpserver (2.x). The API is otherwise identical, so
# import whichever is present rather than pinning one SDK generation.
try:
    from mcp.server.fastmcp import FastMCP as _ServerClass  # SDK 1.x
except ImportError:  # pragma: no cover - depends on the installed SDK
    from mcp.server.mcpserver import MCPServer as _ServerClass  # SDK 2.x

from medexplain.logging_config import audit, get_logger
from agents.retrieval.retriever import get_retrieval_agent

logger = get_logger("MedExplain.MCP.Server")

server = _ServerClass("medexplain-retrieval")


@server.tool()
def retrieve_medical_context(analyte: str, flag: Optional[str] = None, top_k: int = 3) -> str:
    """
    Retrieve trusted medical reference passages about a laboratory analyte.

    Args:
        analyte: Name of the lab test, e.g. "Hemoglobin", "HbA1c", "LDL Cholesterol".
        flag: Optional result flag - "HIGH", "LOW" or "NORMAL" - used to focus the query.
        top_k: Maximum number of passages to return (default 3).

    Returns:
        JSON string with `passages`, `citations`, `grounded` and `backend`.
        `grounded` is false when nothing relevant was found, which means the
        caller must abstain rather than answer from model memory.
    """
    agent = get_retrieval_agent()
    result = agent.retrieve(analyte=analyte, flag=flag, top_k=top_k)

    audit(
        "mcp.tool_call",
        tool="retrieve_medical_context",
        arguments={"analyte": analyte, "flag": flag, "top_k": top_k},
        grounded=result["grounded"],
        passage_ids=[p["id"] for p in result["passages"]],
    )
    logger.info(
        "MCP tool call: retrieve_medical_context(analyte=%r, flag=%r) -> %d passages",
        analyte,
        flag,
        len(result["passages"]),
    )
    return json.dumps(result, ensure_ascii=False)


@server.tool()
def knowledge_base_info() -> str:
    """Report which retrieval backend is active and how many passages are indexed."""
    agent = get_retrieval_agent()
    info: Dict[str, Any] = {
        "backend": agent.backend_name,
        "indexed_passages": agent.store.count(),
        "corpus_path": str(agent.corpus_path),
    }
    logger.info("MCP tool call: knowledge_base_info -> %s", info)
    return json.dumps(info, ensure_ascii=False)


def _list_tools() -> None:
    """Print the exposed tool names - useful as demo evidence."""
    tools: List[str] = ["retrieve_medical_context", "knowledge_base_info"]
    agent = get_retrieval_agent()
    print(json.dumps(
        {
            "server": "medexplain-retrieval",
            "protocol": "Model Context Protocol (MCP)",
            "transport": "stdio",
            "tools": tools,
            "backend": agent.backend_name,
            "indexed_passages": agent.store.count(),
        },
        indent=2,
    ))


if __name__ == "__main__":
    if "--list-tools" in sys.argv:
        _list_tools()
    else:
        logger.info("Starting MedExplain retrieval MCP server on stdio transport")
        server.run(transport="stdio")
