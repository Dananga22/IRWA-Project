"""Agent 3 - RAG Retrieval Agent, exposed to other agents over MCP."""

from agents.retrieval.retriever import RetrievalAgent, get_retrieval_agent

__all__ = ["RetrievalAgent", "get_retrieval_agent"]
