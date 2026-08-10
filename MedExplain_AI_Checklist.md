# MedExplain AI — Group Project Master Checklist & Roadmap

> **Course**: Information Retrieval and Web Analytics (IT 3041)  
> **Lecturer**: Mr. Samadhi Chathuranga Rathnayake  
> **Project Title**: MedExplain AI — Agentic AI Medical Report Explanation System  

---

## Phase 0: Setup & Team Onboarding
- [ ] Register group on LMS (Deadline: End of Week 1)
- [ ] Confirm domain/topic (*Medical Lab Report Analysis & Explanation*) with lecturer
- [ ] Create shared GitHub repository
- [ ] Assign preliminary team roles:
  - **Member 1 (Prompt Injection & Security)**: LLM/Prompting Agent & Prompt Security Audit
  - **Member 2 (Privacy & DB/Auth)**: PDF/NLP extraction, Database & Privacy Audit
  - **Member 3 (Responsible AI & Frontend)**: UI/UX (React), Responsible AI & Bias Audit
  - **Member 4 (IR/RAG & Multi-Agent MCP)**: RAG Pipeline, Vector DB, MCP Protocol & IR Security Audit
- [ ] Each member obtains a free Gemini API key ([Google AI Studio](https://aistudio.google.com/app/apikey))

---

## Phase 1: Minimum Working Pipeline (Proof of Concept)
- [ ] **Step 1 — One LLM Call**: Hardcoded lab test input -> Gemini API explanation output -> Printed to console
- [ ] **Step 2 — PDF Extraction**: PyMuPDF (`fitz`) pulls raw text from sample lab report PDFs
- [ ] **Step 3 — Basic NLP Extraction**: Regex/NER parses test name, value, reference range, and flag (high/low/normal)
- [ ] **Step 4 — End-to-End Pipeline Script**: PDF upload -> Extracted values -> Gemini LLM explanation output
- [ ] **Step 5 — Robustness Testing**: Test with 3-5 different real/sample lab report formats (CBC, Lipid Panel, Metabolic Panel)

---

## Phase 2: Core Usable Application (Web Stack)
- [ ] **Frontend (React / Next.js)**:
  - Patient Upload Page (PDF drop zone, instant preview)
  - Interactive Explanation & Results Dashboard
- [ ] **Backend (FastAPI)**:
  - Upload API endpoint (`/api/v1/reports/upload`) connecting PDF parser to Python engine
  - Report management endpoints
- [ ] **Database (MySQL)**:
  - Schema for Users, Uploaded Reports, Extracted Test Items, and Generated Explanations
- [ ] **Authentication & Security**:
  - JWT Authentication (Patient Login / Registration)
  - Encrypted store for user credentials (Bcrypt)

---

## Phase 3: Information Retrieval & RAG Pipeline
- [ ] Collect 30–50 authoritative medical reference documents (WHO, CDC, MedlinePlus, NIH)
- [ ] Text cleaning & chunking strategy (Recursive character text splitter with metadata preservation)
- [ ] Vector Embeddings generation using Sentence Transformers (`all-MiniLM-L6-v2`)
- [ ] Vector Storage: Store embeddings in ChromaDB / FAISS
- [ ] Upgrade LLM Explanation Agent to retrieve medical context before response generation
- [ ] Implement source citation in explanation output (e.g., "[Source: WHO Blood Guidelines, 2024]")

---

## Phase 4: Multi-Agent Orchestration & Protocol
- [ ] Restructure system using **LangGraph** with a shared state object (`MedicalReportState`)
- [ ] Implement Agent Communication Protocol: Expose RAG Retrieval Agent via **Model Context Protocol (MCP)**
- [ ] Implement **Safety Verification Agent**: Final safety gate checking for non-diagnostic disclaimers, hallucination prevention, and toxic content before output is delivered

---

## Phase 5: Responsible AI & Security Hardening
- [ ] **Input Sanitization**: File type/size validation, prompt injection sanitization before LLM invocation
- [ ] **Encryption & Privacy**: HTTPS transport encryption, sensitive PII database column encryption at rest
- [ ] **Role-Based Access Control (RBAC)**: Distinct permissions for Patient vs. Admin
- [ ] **Responsible AI Pillars**:
  - *Fairness*: Evaluation across non-English query scenarios (stretch target: Sinhala/Tamil)
  - *Explainability*: Cited authoritative sources for every health explanation
  - *Transparency*: Prominent "Not a Medical Diagnosis" disclaimers on every UI screen
  - *Data Protection*: PII redactor agent/filter

---

## Phase 6: Admin Portal (Stretch Goal)
- [ ] Admin authentication flow
- [ ] Admin analytics dashboard (active users, total processed reports, flagged safety responses)
- [ ] Knowledge Base Management Interface (Upload, update, re-index RAG medical documents)

---

## Phase 7: Deliverables & Milestone Schedule
- [ ] **Week 6 — Mid Evaluation (20 Marks)**:
  - System architecture diagram
  - Agent roles & communication flow diagram (LangGraph + MCP)
  - Working progress demonstration
  - Responsible AI compliance check
  - Commercialization pitch
- [ ] **Week 10 — Deliverables Package**:
  - **Gen AI Video (25 Marks)**: 3-5 minute explainer video using AI tools (Synthesia / HeyGen / Pika)
  - **Final Report (30 Marks)**: Complete system design, methodology, Responsible AI implementation, commercialization plan with exact pricing model, and evaluation results
  - **GitHub Repository (5 Marks)**: Clean codebase, comprehensive `README.md` (setup, API docs, contributors)
- [ ] **Week 11 — Group Viva (20 Marks)**:
  - Full-team defense of architecture, agent protocols, Responsible AI, and business model

---

## Individual Security Assessment Alignment (IT 3041)
> *Note: Each team member will act as Red Team Member for the system in Week 10+.*

1. **Student 1 (Prompt Injection & Jailbreak)**: 15 test cases on prompt injection, instruction override, prompt leakage.
2. **Student 2 (Privacy & Data Leakage)**: 15 test cases on PII leakage, conversation memory leakage, auth weaknesses.
3. **Student 3 (Responsible AI & Bias)**: 15 test cases on hallucinations, bias, toxic responses, medical safety boundaries.
4. **Student 4 (IR & Security Assessment)**: 15 test cases on retrieval accuracy, RAG manipulation, API security, MCP communication safety.

---

## Commercialization & Business Strategy
- **Target Market**: Outpatient clinics, lab diagnostic centers, telehealth providers, and health-conscious consumers.
- **B2B Model**: Monthly per-provider tier for hospitals/labs ($199/month/clinic up to 1,000 reports/mo).
- **B2C Premium Tier**: $9.99/month for unlimited patient report uploads, historical trend tracking, and family profile management.
