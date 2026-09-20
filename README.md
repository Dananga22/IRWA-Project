# MedExplain AI

An agentic, multi-agent AI framework for explaining complex medical laboratory reports in plain language, fully grounded in WHO, CDC, and MedlinePlus sources, with full Sinhala localization and strict Responsible AI safety verification gates.

**It explains. It does not diagnose.** This constraint is enforced in code by an automated Safety Verification Agent, not merely by a prompt instruction.

> **Course:** IT3041 — Information Retrieval and Web Analytics  
> **Evaluation:** University Mid-Evaluation & Final Benchmark  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [7-Agent Pipeline](#7-agent-pipeline)
  - [Agent Communication Protocol](#agent-communication-protocol)
  - [Responsible AI \& Safety Verification Gate](#responsible-ai--safety-verification-gate)
- [Security Features](#security-features)
  - [Role-Based Access Control (RBAC)](#role-based-access-control-rbac)
  - [Field-Level AES-256 Encryption at Rest](#field-level-aes-256-encryption-at-rest)
  - [Numerical Fidelity Verification (Sinhala)](#numerical-fidelity-verification-sinhala)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation \& Key Setup](#installation--key-setup)
  - [Database Encryption Migration](#database-encryption-migration)
  - [Seeding Demo Accounts](#seeding-demo-accounts)
  - [Running the Web Application](#running-the-web-application)
- [Usage Guide](#usage-guide)
  - [Patient Flow](#patient-flow)
  - [Lab Assistant Flow](#lab-assistant-flow)
- [Running Tests](#running-tests)
- [Known Limitations](#known-limitations)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview

MedExplain AI bridges the communication gap between clinical laboratory data and patients. When a user uploads a medical laboratory PDF report (e.g., Complete Blood Count, Lipid Panel, Metabolic Panel), the system parses raw numerical data, retrieves trusted medical literature, synthesizes plain-language educational explanations, enforces Responsible AI safety rules, and offers faithful Sinhala localizations alongside general lifestyle care guidance.

The application serves two distinct user roles through dedicated portals:
1. **Patient Dashboard:** A modern, mobile-friendly interface where patients can view their historical lab reports, read AI-generated plain-language summaries, switch between English and Sinhala, and explore lifestyle Care Guides.
2. **Lab Assistant Portal:** An administrative workspace allowing authorized lab staff to upload new reports (via PDF extraction or structured manual entry), manage active records with live search/filtering, perform hard record deletions, and generate instant demo preview sessions for patient walkthroughs.

---

## Architecture

### 7-Agent Pipeline

The core intelligence of MedExplain AI is structured as a 7-agent pipeline:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 LangGraph Shared State                   │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
  Uploaded PDF ──► [Agent 1: Document Processor] (PyMuPDF text extraction)
                                               │
                                               ▼
                   [Agent 2: NLP Entity Extractor] (Regex & noise-filtered entity extraction)
                                               │
                                               ▼
                   [Agent 3: RAG Retrieval Agent] (BM25 / ChromaDB vector search over WHO/CDC/MedlinePlus)
                                               │  ▲
                                  MCP Protocol │  │ Tool Call: retrieve_medical_context
                                               ▼  │
                   [Agent 4: LLM Explanation Agent] (gemini-3.6-flash / Grounded Fallback Explainer)
                                               │
                                               ▼
                   [Agent 5: Safety Verification Agent] ──► (Approved = False?)
                                               │                    │
                                               │  Revision Loop     │ Max 2 retries
                                               │  ◄─────────────────┘
                                               ▼ (Approved = True)
                                  English Explanation & Citations
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
          [Agent 6: Localisation Agent]             [Agent 7: Lifestyle Care Guide Agent]
       (Sinhala translation + numerical check)     (Actionable tips + doctor questions)
```

| Agent Name | Module File | Description |
|---|---|---|
| **Agent 1: Document Processing Agent** | [`agents/document/processor.py`](agents/document/processor.py) | Extracts raw text from uploaded lab report PDFs using PyMuPDF (`fitz`) and handles document sanitization. |
| **Agent 2: NLP Extraction Agent** | [`agents/extraction/entity_extractor.py`](agents/extraction/entity_extractor.py) | Parses analytes, numeric values, units, reference ranges, and flags (`HIGH`/`LOW`/`NORMAL`) via deterministic regex and filters administrative noise. |
| **Agent 3: RAG Retrieval Agent** | [`agents/retrieval/retriever.py`](agents/retrieval/retriever.py) | Queries the curated WHO/CDC/MedlinePlus knowledge base using BM25 or ChromaDB vector embeddings (`all-MiniLM-L6-v2`), configurable via `VECTOR_BACKEND`. |
| **Agent 4: LLM Explanation Agent** | [`agents/explanation/grounded_explainer.py`](agents/explanation/grounded_explainer.py) | Generates patient-friendly explanations strictly cited from retrieved passages using `gemini-3.6-flash` (or a deterministic grounded fallback generator). |
| **Agent 5: Safety Verification Agent** | [`agents/safety/verifier.py`](agents/safety/verifier.py) | Responsible AI control point that audits draft explanations against 5 safety checks before releasing output to the patient. |
| **Agent 6: Localisation Agent** | [`agents/translation/translator.py`](agents/translation/translator.py) | Translates approved English explanations into Sinhala with strict numerical fidelity verification. |
| **Agent 7: Lifestyle Care Guide Agent** | [`agents/care/guide.py`](agents/care/guide.py) | Builds grounded lifestyle recommendations across 5 categories (nutrition, exercise, monitoring, warning signs, doctor questions) for non-critical results. |

### Agent Communication Protocol

MedExplain AI combines three communication protocols:
* **LangGraph Shared State ([`medexplain/graph.py`](medexplain/graph.py)):** Graph nodes read and mutate a typed `MedicalReportState` dictionary. The edge from Agent 5 back to Agent 4 forms a feedback loop allowing up to 2 revision attempts if a draft fails safety verification.
* **Model Context Protocol (MCP) ([`agents/retrieval/mcp_server.py`](agents/retrieval/mcp_server.py) & [`agents/retrieval/mcp_client.py`](agents/retrieval/mcp_client.py)):** Agent 3 is exposed as an MCP tool (`retrieve_medical_context`). Agent 4 dynamically invokes this tool via stdio/inprocess RPC to perform context lookups on demand.
* **REST over HTTPS:** Communicates between the FastAPI backend and the client browser with JWT/session state.

### Responsible AI & Safety Verification Gate

Agent 5 (`SafetyVerificationAgent`) acts as the final decision gate. It executes 5 mandatory checks:
1. **Groundedness Check:** Enforces a minimum 50% lexical overlap between generated claims and retrieved source passages (`GROUNDEDNESS_THRESHOLD = 0.5`).
2. **Scope Check:** Rejects any output that attempts to state a medical diagnosis, name or prescribe medications, or predict prognosis.
3. **Tone & Harm Check:** Rejects alarming, fatalistic, or dismissive language.
4. **Disclaimer Integrity Check:** Ensures the non-diagnostic educational notice remains intact.
5. **Critical-Value Escalation:** Immediately short-circuits evaluation if panic-level lab values are detected, replacing explanations with an urgent medical attention recommendation.

If any check fails, `approved` is set to `False` with detailed revision hints, routing control back to Agent 4 for revision. If revisions fail, the pipeline fails closed and withhold explanations.

---

## Security Features

### Role-Based Access Control (RBAC)
* **User Roles:** `patient` and `lab_assistant`.
* **Access Control:** Patients are strictly isolated to their own records. Lab assistants can upload and manage lab reports but are strictly blocked from viewing AI patient summaries.
* **Password Hashing:** Passwords are hashed using `bcrypt` with unique salts.
* **Rate Limiting:** Login attempts are rate-limited to a maximum of 5 failed attempts per 15-minute window per IP to prevent brute-force attacks (`medexplain/auth.py`).

### Field-Level AES-256 Encryption at Rest
* **AES-256 Cipher:** Sensitive columns in SQLite (`test_type`, `raw_results`, `ai_summary` in `lab_reports`) are encrypted using Fernet (AES-256-CBC with HMAC authentication) via [`medexplain/crypto.py`](medexplain/crypto.py).
* **Strict Key Management:** The encryption key is loaded exclusively from the `MEDEXPLAIN_DB_KEY` environment variable. If missing, the application fails loudly at startup rather than running unencrypted.
* **Metadata Integrity:** Non-sensitive columns (`id`, `patient_id`, `trace_id`, `created_at`) remain unencrypted to support database indexing, sorting, and filtering.

### Numerical Fidelity Verification (Sinhala)
* **Translation Guard:** When Agent 6 translates an explanation into Sinhala, [`verify_translation()`](agents/translation/translator.py) extracts every numeric value and range.
* **Tamper Prevention:** If a number is altered, rounded, or omitted during translation (e.g. `245.0` translated as `445.0`), the translation is instantly rejected (`verified = False`) and withheld from the user.

### HTTPS / TLS Encryption in Transit
* **Local Demonstration:** Includes [`scripts/generate_cert.py`](scripts/generate_cert.py) to generate self-signed TLS certificates (`certs/cert.pem`, `certs/key.pem`) for local `https://` demonstrations and viva evaluations (`USE_SSL=1 python app.py` or Uvicorn `--ssl-keyfile`).
* **Production Deployment:** In a production deployment, TLS termination would sit behind a reverse proxy (e.g., Nginx, Caddy) or managed cloud TLS termination (e.g., AWS ALB, Cloudflare, GCP Cloud Run). The self-signed certificates provided here are strictly for local demonstration purposes.

---

## Tech Stack

* **Language:** Python 3.9+
* **Web Framework:** FastAPI, Uvicorn, Python-Multipart
* **Agent Orchestration & Communication:** LangGraph, LangChain-Core, Model Context Protocol (`mcp` SDK)
* **LLM Integration:** Google GenAI SDK (`google-genai`), `gemini-3.6-flash` (configurable via `GEMINI_MODEL`, with automatic fallback to a deterministic grounded generator when no API key is provided)
* **Search & Vector Database:** Configurable via `VECTOR_BACKEND` (`auto`, `chromadb`, or `bm25`). Defaults to `auto` (uses ChromaDB with `all-MiniLM-L6-v2` / ONNX embeddings if installed, with automatic fallback to an in-memory BM25 index)
* **PDF Processing & Security:** PyMuPDF (`fitz`), Cryptography (`cryptography.fernet`), Bcrypt
* **Persistence:** SQLite3 (`data/medexplain.db` with AES-256 field-level encryption at rest)
* **Frontend:** HTML5, Vanilla CSS3 (Dark Navy theme, glassmorphism, responsive grid), Vanilla JavaScript (ES6+ Fetch API)

---

## Setup Instructions

### Prerequisites
* Python 3.9 or higher
* Git

### Installation & Key Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/medexplain-ai.git
   cd medexplain-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```

5. **Generate a fresh database encryption key:**
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```
   Copy the generated string and update `.env`:
   ```env
   MEDEXPLAIN_DB_KEY=your_generated_fernet_key_here
   GEMINI_API_KEY=your_gemini_api_key_here  # Optional: falls back to grounded generator if unset
   ```

### Database Encryption Migration

If setting up from an existing SQLite database containing unencrypted legacy test records, run the idempotent migration script to encrypt all clinical values at rest:

```bash
python scripts/migrate_encrypt_db.py
```

### Seeding Demo Accounts

To create temporary local development accounts (one patient, one lab assistant) with randomly generated passwords printed to your console, run:

```bash
python scripts/seed_demo_users.py
```

---

## Running the Application

### HTTP Mode (Default)
```bash
python app.py
```

### HTTPS / TLS Mode (Local Demonstration & Viva)
1. **Generate self-signed TLS certificates:**
   ```bash
   python scripts/generate_cert.py
   ```
2. **Start server over HTTPS:**
   ```bash
   # Option A: via environment variable
   USE_SSL=1 python app.py

   # Option B: via Uvicorn CLI
   uvicorn app:app --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --port 8000
   ```

---

## Usage Guide

*Note: Demo accounts generated via `scripts/seed_demo_users.py` are throwaway credentials strictly for local development and testing, never used in any production deployment.*

### Patient Flow
1. **Login:** Generate demo credentials with `python scripts/seed_demo_users.py` and log into the Patient Portal.
2. **View Reports:** Select a report from the historical lab report list.
3. **AI Summary:** Read plain-language educational explanations for each lab analyte alongside citations to MedlinePlus/WHO.
4. **Care Guide:** Click **Open Care Guide** to view lifestyle recommendations covering nutrition, exercise, monitoring, warning signs, and questions for your doctor.
5. **Language Switch:** Toggle between **English** and **සිංහල (Sinhala)** to read verified translations.

### Lab Assistant Flow
1. **Login:** Log into the Lab Assistant Portal using lab assistant demo credentials generated via `python scripts/seed_demo_users.py`.
2. **Upload Results:** Upload a patient report PDF for automated entity extraction or fill out the structured manual lab result form.
3. **Manage Records:** Filter, search, and delete duplicate or obsolete lab records.
4. **Demo Preview:** Click **Demo to Patient** to open an ephemeral patient preview without creating permanent database entries.

---

## Running Tests

MedExplain AI includes 14 unit test suites covering authentication, PDF extraction, RAG retrieval, LLM explanation, safety verification, translation fidelity, care guides, and field-level encryption.

Run the entire test suite using Python `unittest`:

```bash
python -m unittest tests.test_auth tests.test_care_guide tests.test_crypto tests.test_diverse_reports tests.test_document tests.test_extraction tests.test_extraction_filter tests.test_graph tests.test_llm tests.test_result_store tests.test_retrieval tests.test_safety tests.test_summarisation tests.test_translation
```

**Current Test Status:** `145 tests passing` (100% pass rate across all 14 test suites).

---

## Known Limitations

1. **Knowledge Base Scope:** The RAG retrieval corpus (`knowledge_base/corpus.json`) contains 59 passages covering primary blood panels (CBC, Lipid, Metabolic, HbA1c, Thyroid, Renal). For rare or unindexed analytes, the system intentionally **abstains** from generating an explanation rather than hallucinating medical facts.
2. **Tamil Localization Exposure:** Backend validation dictionaries and script checks support Tamil (`ta`), but the current user interface modal exposes an English / Sinhala toggle button (`en` / `si`).

---

## Contributors

| Name | Role | Specialization |
|---|---|---|
| **[Student Name]** | Lead Developer / Architect | Agentic Pipelines & RAG Retrieval |
| **[Student Name]** | Security & Backend Engineer | Cryptography, Auth & RBAC |
| **[Student Name]** | AI & NLP Engineer | Entity Extraction & Responsible AI Safety |
| **[Student Name]** | Frontend & UX Engineer | Patient Portal & UI/UX |

---

## License

Distributed under the MIT License. See `LICENSE` for more information.  
Developed for academic evaluation under course IT3041 (Information Retrieval & Web Analytics).
