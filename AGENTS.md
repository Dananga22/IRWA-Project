# AGENTS.md — Standing Instructions for MedExplain AI

## PROJECT OVERVIEW
MedExplain AI is a multi-agent system that reads a patient's uploaded medical report (blood test PDF), extracts test values, looks up trusted medical information about them (RAG), and writes a plain-English explanation. University project for IT3041 Information Retrieval and Web Analytics.

## HARD RULES THE SYSTEM MUST FOLLOW
1. **Never diagnose**: Never name a disease as a clinical conclusion.
2. **Never prescribe**: Never recommend, name, or dose a medication.
3. **Never alter treatment**: Never tell a user to stop or change a medical treatment.
4. **Grounded citations**: Every medical claim in output must cite a retrieved source passage.
5. **Physician consultation**: Always tell the user to discuss results with their doctor.

## TECH STACK
- Language: Python 3.11+
- Frameworks: FastAPI, React
- Database: MySQL (with SQLite fallback)
- PDF Extraction: PyMuPDF (fitz)
- Embeddings & Vector Search: sentence-transformers, ChromaDB
- Orchestration & Protocol: LangChain / LangGraph, Model Context Protocol (MCP)
- LLM API: Google Gemini API

## CODING RULES
- **Type hints**: Required on every function signature.
- **Testing**: Every module gets a matching test file in `tests/`.
- **Secrets Management**: No secrets in code; all configuration via environment variables (`.env`).
- **Logging**: No `print()` for logging — use standard `logging` module.
- **Modular Agents**: Keep each agent in its own directory under `agents/` or `medexplain/agents/`.
