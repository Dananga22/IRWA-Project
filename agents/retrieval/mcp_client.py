"""
MedExplain AI - MCP client used by the LLM Explanation Agent.
Module: agents.retrieval.mcp_client

Two transports, one tool contract:

  stdio     - spawns agents/retrieval/mcp_server.py and speaks real MCP
              JSON-RPC over stdio. This is the protocol demonstration.
  inprocess - calls the retrieval agent directly, keeping the identical tool
              name and argument schema. Used when the MCP SDK is unavailable.

Select with MCP_TRANSPORT=stdio | inprocess | auto (default auto).

Two details that matter in practice:

  * All analytes for one report are sent through a single MCP session
    (`call_tools`), so a report costs one subprocess rather than one per test.
  * FastAPI runs inside an event loop, where asyncio.run() cannot be called.
    The stdio path is therefore executed on a worker thread, so the protocol
    works identically from the CLI and from the web API.

Every call is written to the audit log on either transport, so the agent hop is
visible in the trace regardless of which one is active.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from medexplain.logging_config import audit, get_logger

logger = get_logger("MedExplain.MCP.Client")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_NAME = "retrieve_medical_context"

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-stdio")

# The server subprocess imports the retrieval agent, which loads a sentence
# embedding model. The first load can take a minute (the model downloads), and
# the SDK's default timeout is far shorter - which surfaced as an opaque
# "unhandled errors in a TaskGroup" and a silent fall back to in-process.
MCP_TIMEOUT_SECONDS = float(os.getenv("MCP_TIMEOUT_SECONDS", "180"))

# The subprocess's stderr is captured so a startup failure names itself instead
# of disappearing.
STDERR_LOG = Path(os.getenv("MEDEXPLAIN_LOG_DIR", "logs")) / "mcp_server.log"


class MCPRetrievalClient:
    """Client wrapper around the retrieval tool exposed over MCP."""

    def __init__(self, transport: Optional[str] = None) -> None:
        requested = (transport or os.getenv("MCP_TRANSPORT", "auto")).lower()
        self.transport = self._resolve_transport(requested)
        logger.info("MCP retrieval client using transport: %s", self.transport)

    @staticmethod
    def _resolve_transport(requested: str) -> str:
        if requested == "inprocess":
            return "inprocess"
        try:
            import mcp  # noqa: F401
        except ImportError:
            if requested == "stdio":
                logger.warning("MCP SDK not installed; falling back to in-process transport.")
            return "inprocess"
        return "stdio" if requested in ("stdio", "auto") else "inprocess"

    # ------------------------------------------------------------------ public

    def call_tool(self, analyte: str, flag: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """Invoke retrieve_medical_context once."""
        return self.call_tools([{"analyte": analyte, "flag": flag, "top_k": top_k}])[0]

    def call_tools(self, requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Invoke retrieve_medical_context for several analytes.

        On the stdio transport all calls share one MCP session, so a report
        costs a single server process rather than one per test.
        """
        if not requests:
            return []

        audit(
            "mcp.client_request",
            tool=TOOL_NAME,
            transport=self.transport,
            call_count=len(requests),
            analytes=[r.get("analyte") for r in requests],
        )

        if self.transport == "stdio":
            try:
                return self._run_blocking(self._call_over_stdio(list(requests)))
            except Exception as exc:
                logger.warning("MCP stdio session failed (%s). Falling back to in-process transport.", exc)
                logger.warning("  server stderr: %s", _tail_server_log())
                logger.warning(
                    "  if this is a first run, the embedding model may still be downloading; "
                    "raise MCP_TIMEOUT_SECONDS or set MCP_TRANSPORT=inprocess"
                )
                audit(
                    "mcp.transport_fallback",
                    tool=TOOL_NAME,
                    error=str(exc),
                    server_stderr=_tail_server_log(),
                )

        return [self._call_in_process(r) for r in requests]

    # ----------------------------------------------------------------- private

    @staticmethod
    def _run_blocking(coro: Any) -> List[Dict[str, Any]]:
        """
        Run an async MCP session from sync code, in or out of an event loop.

        FastAPI handlers already run inside a loop, where asyncio.run() raises;
        there the coroutine is executed on a dedicated worker thread instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)  # plain sync context, e.g. the CLI runner
        return _EXECUTOR.submit(asyncio.run, coro).result()

    @staticmethod
    def _call_in_process(arguments: Dict[str, Any]) -> Dict[str, Any]:
        from agents.retrieval.retriever import get_retrieval_agent

        agent = get_retrieval_agent()
        result = agent.retrieve(
            analyte=arguments["analyte"],
            flag=arguments.get("flag"),
            top_k=arguments.get("top_k", 3),
        )
        result["transport"] = "inprocess"
        return result

    @staticmethod
    async def _call_over_stdio(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agents.retrieval.mcp_server"],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )

        results: List[Dict[str, Any]] = []
        STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with STDERR_LOG.open("w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=MCP_TIMEOUT_SECONDS),
                ) as session:
                    await session.initialize()

                    for arguments in requests:
                        response = await session.call_tool(TOOL_NAME, arguments)
                        payload = "".join(
                            block.text for block in response.content
                            if getattr(block, "type", None) == "text"
                        )
                        result = json.loads(payload) if payload else {
                            "analyte": arguments.get("analyte"),
                            "passages": [],
                            "citations": [],
                            "grounded": False,
                        }
                        result["transport"] = "stdio"
                        results.append(result)

        return results


def _tail_server_log(lines: int = 6) -> str:
    """Last few lines of the MCP server's stderr, for diagnosing a failed start."""
    try:
        content = STDERR_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return "(no server log)"
    return " | ".join(content[-lines:]) if content else "(server produced no output)"


_CLIENT_SINGLETON: Optional[MCPRetrievalClient] = None


def get_mcp_client() -> MCPRetrievalClient:
    """Process-wide singleton client."""
    global _CLIENT_SINGLETON
    if _CLIENT_SINGLETON is None:
        _CLIENT_SINGLETON = MCPRetrievalClient()
    return _CLIENT_SINGLETON
