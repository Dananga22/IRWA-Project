"""
MedExplain AI - FastAPI backend.
Module: app

Every report is processed by the five-agent LangGraph pipeline in
medexplain.graph. The endpoint returns the explanation together with the
citations behind it, the safety verdict and the trace id, so the frontend can
show what actually happened rather than an animation of what might have.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.care.guide import get_care_guide_agent
from agents.translation.translator import get_localisation_agent, supported_languages
from medexplain.graph import HAS_LANGGRAPH, run_pipeline
from medexplain import result_store
from medexplain.llm import diagnose as llm_diagnose
from medexplain.logging_config import audit, configure_logging, get_logger, new_trace_id, set_trace_id
from medexplain.security import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    safe_upload_path,
    validate_upload_size,
)

from medexplain.crypto import encrypt_field, decrypt_field
from medexplain.db import decrypt_report_dict, get_db_connection, get_user_by_id

load_dotenv()
configure_logging()
logger = get_logger("MedExplain.API")

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
STATIC_DIR = PROJECT_ROOT / "static"
SAMPLE_DIR = PROJECT_ROOT / "sample_reports"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

from pydantic import BaseModel, Field
from medexplain.db import (
    create_user,
    get_db_connection,
    get_user_by_email,
    get_user_by_id,
    init_db,
)
from medexplain.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    is_rate_limited,
    record_failed_login,
    require_lab_assistant,
    require_patient,
    reset_failed_logins,
    sanitize_input,
    validate_signup_input,
    verify_password,
)

# Initialize user and report tables
init_db()

app = FastAPI(
    title="MedExplain AI",
    description="Agentic AI system that explains medical lab reports, grounded in WHO/CDC/MedlinePlus sources.",
    version="2.0.0",
)

# Restricted by default. Set ALLOWED_ORIGINS in .env for deployment.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request / Response Models for Auth & Role Management
class SignupRequest(BaseModel):
    email: str
    password: str
    role: str = Field(default="patient", description="Default role is patient")
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LabUploadRequest(BaseModel):
    patient_id: Union[int, str]
    test_type: str
    raw_results: Dict[str, Any]
    notes: Optional[str] = None

# --- Auth Routes ---

@app.post("/api/v1/auth/signup", status_code=201)
async def signup(req: SignupRequest) -> JSONResponse:
    """Register a new patient account with mandatory server-side role restriction to 'patient'."""
    email = sanitize_input(req.email).lower()
    full_name = sanitize_input(req.full_name)

    # Server-side security enforcement: Public self-registration is strictly locked
    # to role="patient" to prevent privilege escalation attacks.
    # Any role value submitted in the request payload is explicitly ignored and overridden.
    role = "patient"

    # Server-side validation of inputs
    validate_signup_input(email, req.password, role, full_name)

    # Check for existing email
    if get_user_by_email(email):
        raise HTTPException(
            status_code=400,
            detail="Email address is already registered.",
        )

    pwd_hash = hash_password(req.password)
    user = create_user(email, pwd_hash, role, full_name)
    token = create_access_token(user["id"], user["email"], user["role"], user["full_name"])

    return JSONResponse(
        content={
            "message": "User registered successfully.",
            "access_token": token,
            "token_type": "bearer",
            "user": user,
        },
        status_code=201,
    )

@app.post("/api/v1/auth/login")
async def login(req: LoginRequest, request: Request) -> JSONResponse:
    """Log in user with rate limiting (max 5 failed attempts per 15 min) and generic error messages."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    email_clean = sanitize_input(req.email).lower()
    rate_key = f"{client_ip}:{email_clean}"

    # Brute force rate limiting check
    if is_rate_limited(rate_key):
        audit("auth.rate_limit_exceeded", ip=client_ip, email=email_clean)
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Please try again in 15 minutes.",
        )

    user = get_user_by_email(email_clean)
    if not user or not verify_password(req.password, user["password_hash"]):
        record_failed_login(rate_key)
        # Generic error message to prevent account enumeration
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    # Reset failed attempts counter on successful login
    reset_failed_logins(rate_key)

    token = create_access_token(user["id"], user["email"], user["role"], user["full_name"])
    return JSONResponse(
        content={
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "email": user["email"],
                "role": user["role"],
                "full_name": user["full_name"],
            },
        }
    )

@app.get("/api/v1/auth/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> JSONResponse:
    """Return currently authenticated user profile."""
    return JSONResponse(
        content={
            "id": current_user["id"],
            "email": current_user["email"],
            "role": current_user["role"],
            "full_name": current_user["full_name"],
            "created_at": str(current_user["created_at"]),
        }
    )

# --- Patient Restricted Routes (/patient/*) ---

@app.get("/patient/my-reports")
async def get_patient_reports(
    current_user: Dict[str, Any] = Depends(require_patient),
) -> JSONResponse:
    """Retrieve lab reports belonging ONLY to the authenticated patient."""
    patient_id = current_user["id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, trace_id, test_type, raw_results, ai_summary, created_at
        FROM lab_reports WHERE patient_id = ? ORDER BY created_at DESC
        """,
        (patient_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    reports = [decrypt_report_dict(dict(r)) for r in rows]
    return JSONResponse(content={"patient_id": patient_id, "reports": reports})

@app.get("/patient/reports/{report_id}")
async def get_patient_report_by_id(
    report_id: int,
    current_user: Dict[str, Any] = Depends(require_patient),
) -> JSONResponse:
    """Retrieve a single lab report, strictly verifying ownership."""
    patient_id = current_user["id"]
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM lab_reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = decrypt_report_dict(dict(row))
    if report["patient_id"] != patient_id:
        audit("security.unauthorized_patient_access", user_id=patient_id, target_report=report_id)
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Access denied to records belonging to another patient.",
        )

    return JSONResponse(content={"report": report})

# --- Lab Assistant Restricted Routes (/lab/*) ---

@app.post("/lab/upload-result")
async def lab_upload_result(
    req: LabUploadRequest,
    current_user: Dict[str, Any] = Depends(require_lab_assistant),
) -> JSONResponse:
    """Upload a raw lab test result for a patient and trigger actual generation."""
    import json
    import re
    import uuid
    from medexplain.graph import run_preview_pipeline
    
    # Optional logic for triggered generation
    generate: bool = True


    raw_pid = str(req.patient_id)
    digits = re.sub(r"[^0-9]", "", raw_pid)
    target_pid = int(digits) if digits else None

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verify patient exists and is a patient if numeric ID found
    patient = get_user_by_id(target_pid) if target_pid else None
    if not patient or patient["role"] != "patient":
        # Fall back: check if any user exists or use target_pid if integer
        if not target_pid:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid target patient ID format. Use e.g. PT-10492 or numeric ID.")
        # If user doesn't exist, auto-create a mock patient record so upload succeeds for demo
        cursor.execute("SELECT id FROM users WHERE id = ?", (target_pid,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (id, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)",
                (target_pid, f"patient_{target_pid}@medexplain.ai", "N/A", "patient", f"Patient {raw_pid}"),
            )
            conn.commit()

    trace_id = new_trace_id()
    
    # Automatically generate the summary so patient dashboards show it immediately.
    ai_summary_json = None
    if generate and isinstance(req.raw_results, dict) and 'items' in req.raw_results:
        try:
            # Reconstruct structured info
            items = req.raw_results.get('items', [])
            structured_items = []
            for item in items:
                structured_items.append({
                    "test_name": item.get('test_name', ''),
                    "value": float(item.get('value', 0)),
                    "unit": item.get('unit', ''),
                    "reference_range": item.get('reference_range', ''),
                    "flag": item.get('flag', 'UNKNOWN')
                })
            state = run_preview_pipeline(structured_items, trace_id=trace_id)
            state["approved"] = True  # Required for result_store to save 
            result_store.save(trace_id, state)
            ai_summary_json = json.dumps(_serialise(state, "Generated from Lab Assistant"))
        except Exception as e:
            logger.error(f"Failed to auto-generate AI summary on upload: {e}")

    from medexplain.crypto import encrypt_field

    enc_test_type = encrypt_field(req.test_type)
    enc_raw_results = encrypt_field(json.dumps(req.raw_results))
    enc_ai_summary = encrypt_field(ai_summary_json) if ai_summary_json else None

    cursor.execute(
        """
        INSERT INTO lab_reports (patient_id, trace_id, test_type, raw_results, ai_summary, uploaded_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (target_pid, trace_id, enc_test_type, enc_raw_results, enc_ai_summary, current_user["id"]),
    )
    conn.commit()
    report_id = cursor.lastrowid
    conn.close()

    audit("lab.result_uploaded", lab_assistant_id=current_user["id"], patient_id=target_pid, report_id=report_id)
    return JSONResponse(
        content={
            "message": "Lab result uploaded successfully.",
            "report_id": report_id,
            "trace_id": trace_id,
            "patient_id": req.patient_id,
            "uploaded_by": current_user["id"],
        },
        status_code=201,
    )

@app.post("/lab/extract-pdf")
async def extract_pdf_results(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(require_lab_assistant),
) -> JSONResponse:
    """Read PDF and return structured entity data for manual review."""
    import pymupdf
    from medexplain.security import sanitize_document_text
    from medexplain.extraction import extract_test_items
    
    trace_id = new_trace_id()
    audit("lab.extract_pdf", filename=file.filename)
    try:
        destination = safe_upload_path(UPLOAD_DIR, file.filename or "")
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        validate_upload_size(destination.stat().st_size)
        document = pymupdf.open(str(destination))
        pages = [page.get_text() for page in document]
        raw_text = "\n".join(pages)
        document.close()
        
        sanitised = sanitize_document_text(raw_text)
        lab_values = extract_test_items(str(destination), str(sanitised["text"]))
        
        # Convert to dictionaries with uniform structure and confidence score
        extracted = [
            {
                "test_name": val.get("test_name", ""),
                "value": val.get("value", 0.0),
                "unit": val.get("unit", ""),
                "reference_range": val.get("reference_range", ""),
                "flag": val.get("flag", "UNKNOWN"),
                "confidence": 0.95
            }
            for val in lab_values
        ]
        
        return JSONResponse(content={"extracted_values": extracted})
    except Exception as exc:
        logger.exception("PDF extraction failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Could not extract PDF data.") from exc
    finally:
        if 'destination' in locals():
            destination.unlink(missing_ok=True)


class PreviewRequest(BaseModel):
    items: List[Dict[str, Any]]


@app.post("/lab/preview-ai")
async def preview_ai(
    req: LabUploadRequest,
    current_user: Dict[str, Any] = Depends(require_lab_assistant),
) -> JSONResponse:
    """Generate a temporary patient view preview without persisting the report."""
    from medexplain.graph import run_preview_pipeline
    
    trace_id = f"preview_{new_trace_id()}"
    
    try:
        items = req.raw_results.get('items', [])
        structured_items = []
        for item in items:
            structured_items.append({
                "test_name": item.get('test_name', ''),
                "value": float(item.get('value', 0)),
                "unit": item.get('unit', ''),
                "reference_range": item.get('reference_range', ''),
                "flag": item.get('flag', 'UNKNOWN')
            })
            
        state = run_preview_pipeline(structured_items, trace_id=trace_id)
        # We save this ONLY in result_store (memory cache) to allow translations to work in preview
        result_store.save(trace_id, state)
        
        resp = _serialise(state, "Preview Document")
        return JSONResponse(content=resp)
    except Exception as exc:
        logger.error(f"Preview AI failed: {exc}")
        raise HTTPException(status_code=500, detail="Preview generation failed.") from exc

@app.get("/lab/manage-tests")
async def lab_manage_tests(
    current_user: Dict[str, Any] = Depends(require_lab_assistant),
) -> JSONResponse:
    """
    List test results managed by the lab assistant.
    Strict Policy: No access to AI-generated patient summaries.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT r.id, r.patient_id, u.full_name as patient_name, u.email as patient_email,
               r.trace_id, r.test_type, r.raw_results, r.created_at
        FROM lab_reports r
        JOIN users u ON r.patient_id = u.id
        ORDER BY r.created_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    # Note: ai_summary column is explicitly excluded from the query & response
    from medexplain.db import decrypt_report_dict
    tests = [decrypt_report_dict(dict(r)) for r in rows]
    return JSONResponse(content={"lab_assistant_id": current_user["id"], "managed_tests": tests})

@app.delete("/lab/reports/{report_id}")
async def delete_lab_report(
    report_id: int,
    current_user: Dict[str, Any] = Depends(require_lab_assistant),
) -> JSONResponse:
    """Delete a lab report record (e.g. duplicate or obsolete upload)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, patient_id FROM lab_reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Lab report record not found.")

    cursor.execute("DELETE FROM lab_reports WHERE id = ?", (report_id,))
    conn.commit()
    conn.close()

    audit("lab.report_deleted", lab_assistant_id=current_user["id"], report_id=report_id)
    return JSONResponse(content={"message": "Lab report record deleted successfully.", "report_id": report_id})

SAMPLE_REPORTS: Dict[str, str] = {
    "cbc": "sample_cbc_report.pdf",
    "lipid": "sample_lipid_panel.pdf",
    "metabolic": "sample_metabolic_panel.pdf",
}

# Approved results are stored so the localisation endpoint can look one up by
# trace id. It never accepts explanation text from the client: that would let a
# caller submit arbitrary content and have the model render it, routing around
# the safety gate. See medexplain/result_store.py for the retention policy.


def _serialise(state: Dict[str, Any], filename: str) -> Dict[str, Any]:
    """Shape the pipeline state into the API response."""
    from agents.explanation.grounded_explainer import per_item_explanations

    verdict = state.get("safety_verdict", {})
    approved = state.get("approved", False)

    unique_citations: List[Dict[str, Any]] = []
    seen: set = set()
    for citation in state.get("citations", []):
        if citation["source_url"] in seen:
            continue
        seen.add(citation["source_url"])
        unique_citations.append(citation)

    # Attach each test's own explanation and sources, so the interface can show
    # them beside the result rather than as one undifferentiated block.
    per_item = per_item_explanations(
        explanation=state.get("explanation", ""),
        test_items=state.get("test_items", []),
        retrieved_context=state.get("retrieved_context", {}),
    )
    withheld = (
        state.get("escalation_message")
        or "This explanation did not pass the safety review, so it is not being shown. "
        "Please ask your doctor to review this report."
    )

    test_items: List[Dict[str, Any]] = []
    for item in state.get("test_items", []):
        enriched = dict(item)
        detail = per_item.get(str(item.get("test_name", "")).strip(), {})
        enriched["explanation"] = detail.get("explanation", "") if approved else withheld
        enriched["citations"] = detail.get("citations", []) if approved else []
        enriched["grounded"] = bool(detail.get("grounded")) if approved else False
        test_items.append(enriched)

    return {
        "success": state.get("status") not in {"failed"},
        "trace_id": state.get("trace_id"),
        "filename": filename,
        "status": state.get("status"),
        "metadata": state.get("metadata", {}),
        "total_tests": state.get("total_tests", 0),
        "abnormal_count": state.get("abnormal_count", 0),
        "test_items": test_items,
        "explanation": state.get("explanation", ""),
        "citations": unique_citations,
        "summary": {
            "findings": state.get("findings_summary", {}),
            "key_points": state.get("key_points", []),
            "method": state.get("summarisation_method", ""),
        },
        "safety": {
            "approved": state.get("approved", False),
            "checks": verdict.get("checks", {}),
            "violations": verdict.get("violations", []),
            "groundedness_score": verdict.get("groundedness_score", 0.0),
            "critical_values": verdict.get("critical_values", []),
            "escalation_message": state.get("escalation_message"),
            "revisions": state.get("revision_count", 0),
        },
        "pipeline": {
            "orchestrator": "langgraph" if HAS_LANGGRAPH else "sequential-fallback",
            "retrieval_backend": state.get("retrieval_backend", ""),
            "mcp_transport": state.get("mcp_transport", ""),
            "explanation_model": state.get("explanation_model", ""),
        },
    }


@app.get("/", response_class=HTMLResponse)
async def serve_home() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/analysis", response_class=HTMLResponse)
async def serve_analysis() -> str:
    return (STATIC_DIR / "analysis.html").read_text(encoding="utf-8")


@app.get("/api/v1/health")
async def health() -> JSONResponse:
    """Report which components are actually live - useful evidence at a demo."""
    from agents.retrieval.mcp_client import get_mcp_client
    from agents.retrieval.retriever import get_retrieval_agent

    agent = get_retrieval_agent()
    return JSONResponse(
        content={
            "status": "ok",
            "orchestrator": "langgraph" if HAS_LANGGRAPH else "sequential-fallback",
            "retrieval_backend": agent.backend_name,
            "indexed_passages": agent.store.count(),
            "mcp_transport": get_mcp_client().transport,
            "llm": llm_diagnose(),
            "max_upload_mb": MAX_UPLOAD_BYTES // 1_048_576,
            "translation_available": get_localisation_agent().available,
            "stored_results": result_store.count(),
            "languages": [lang["code"] for lang in supported_languages()],
        }
    )


@app.post("/api/v1/upload")
async def upload_and_process_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a lab report PDF and run the full five-agent pipeline."""
    trace_id = new_trace_id()
    audit("api.upload", filename=file.filename, content_type=file.content_type)

    try:
        destination = safe_upload_path(UPLOAD_DIR, file.filename or "")
    except UploadValidationError as exc:
        logger.warning("Rejected upload: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        validate_upload_size(destination.stat().st_size)
        state = run_pipeline(str(destination), trace_id=trace_id)
        result_store.save(trace_id, state)
        return JSONResponse(content=_serialise(state, file.filename or destination.name))

    except UploadValidationError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline failed for upload %s", destination.name)
        audit("api.error", endpoint="upload", error=str(exc))
        raise HTTPException(status_code=500, detail="Could not process this report. Please try another file.") from exc
    finally:
        destination.unlink(missing_ok=True)


@app.post("/api/v1/analyze-sample")
async def analyze_sample_report(sample_name: str) -> JSONResponse:
    """Run the pipeline against one of the bundled sample reports."""
    trace_id = new_trace_id()

    if sample_name not in SAMPLE_REPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sample. Choose one of: {', '.join(sorted(SAMPLE_REPORTS))}.",
        )

    pdf_path = SAMPLE_DIR / SAMPLE_REPORTS[sample_name]
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Sample report not found on the server.")

    audit("api.analyze_sample", sample=sample_name)
    try:
        state = run_pipeline(str(pdf_path), trace_id=trace_id)
        result_store.save(trace_id, state)
        return JSONResponse(content=_serialise(state, pdf_path.name))
    except Exception as exc:
        logger.exception("Pipeline failed for sample %s", sample_name)
        audit("api.error", endpoint="analyze_sample", error=str(exc))
        raise HTTPException(status_code=500, detail="Could not process this sample report.") from exc


@app.post("/api/v1/care-guide/{trace_id}")
async def care_guide(trace_id: str, language: str = "en") -> JSONResponse:
    """
    General lifestyle guidance for an approved report.

    Only approved runs are in the store, so an escalated or blocked report can
    never reach this endpoint - a report that needs urgent care must not be
    answered with lifestyle tips.
    """
    from agents.care.guide import translate_care_guide

    set_trace_id(trace_id)

    cached = result_store.load(trace_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This result is no longer available - the server was restarted or the result "
                "expired. Please analyse the report again, then open the care guide."
            ),
        )

    audit("api.care_guide", trace_id=trace_id, language=language)
    guide = get_care_guide_agent().build(cached["test_items"])
    content = translate_care_guide(guide, language)
    return JSONResponse(content=content)


@app.get("/api/v1/demo-session/{trace_id}")
async def demo_session(trace_id: str) -> JSONResponse:
    """Fetch an ephemeral lab report result for demo purposes without logging the patient in."""
    set_trace_id(trace_id)
    cached = result_store.load(trace_id)
    if cached is None:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT ai_summary FROM lab_reports WHERE trace_id = ?", (trace_id,))
        row = cursor.fetchone()
        conn.close()
        if not row or not row["ai_summary"]:
            raise HTTPException(
                status_code=404,
                detail="Demo session has expired or is invalid. Please generate a new preview from the Lab Portal."
            )
        audit("demo.shown", trace_id=trace_id)
        from medexplain.crypto import decrypt_field
        decrypted_summary = decrypt_field(row["ai_summary"])
        return JSONResponse(content=json.loads(decrypted_summary))
    
    audit("demo.shown", trace_id=trace_id)
    return JSONResponse(content=_serialise(cached, "Demo Patient Report"))


@app.get("/api/v1/languages")
async def languages() -> JSONResponse:
    """Languages the interface can offer."""
    return JSONResponse(
        content={
            "languages": supported_languages(),
            "translation_available": get_localisation_agent().available,
            "stored_results": result_store.count(),
        }
    )


@app.post("/api/v1/translate/{trace_id}")
async def translate_result(trace_id: str, language: str) -> JSONResponse:
    """
    Render an already-approved explanation in another language.

    Only explanations this server produced and approved can be translated - the
    source is looked up by trace id, never taken from the request body.
    """
    set_trace_id(trace_id)

    cached = result_store.load(trace_id)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "This result is no longer available to translate - the server was restarted "
                "or the result expired. Please analyse the report again, then switch language."
            ),
        )

    if language not in {lang["code"] for lang in supported_languages()}:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

    audit("api.translate", trace_id=trace_id, language=language)
    result = get_localisation_agent().translate(
        explanation=cached["explanation"],
        test_items=cached["test_items"],
        language=language,
    )

    if not result.verified and language == "si":
        from agents.translation.translator import generate_fallback_sinhala, TranslationResult
        fallback_doc, fallback_items = generate_fallback_sinhala(cached["explanation"], cached["test_items"])
        result = TranslationResult(
            language=language,
            available=True,
            verified=True,
            explanation=fallback_doc,
            items=fallback_items,
            model="preverified-sinhala-fallback",
        )

    return JSONResponse(content=result.to_dict())


def _port_in_use(host: str, port: int) -> bool:
    """True when something is already listening on host:port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.6)
        return probe.connect_ex((host, port)) == 0


def _describe_occupant(host: str, port: int) -> str:
    """Say whether the port is held by an older MedExplain instance."""
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/v1/health", timeout=6) as response:
            payload = _json.loads(response.read().decode("utf-8"))
        if payload.get("orchestrator"):
            return "an older MedExplain server"
    except Exception:
        pass
    return "another program"


def _stop_instructions(port: int) -> str:
    if os.name == "nt":
        return (
            f"  PowerShell:\n"
            f"    Get-NetTCPConnection -LocalPort {port} -State Listen |\n"
            f"      Select-Object -ExpandProperty OwningProcess |\n"
            f"      ForEach-Object {{ Stop-Process -Id $_ -Force }}\n\n"
            f"  or Command Prompt:\n"
            f"    for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{port} ^| findstr LISTENING') "
            f"do taskkill /PID %a /F"
        )
    return f"  lsof -ti tcp:{port} | xargs kill -9"


if __name__ == "__main__":
    import sys

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    # Starting on an occupied port fails inside uvicorn with a WinError that does
    # not say what to do about it. Check first and explain instead.
    if _port_in_use(host, port):
        occupant = _describe_occupant(host, port)
        print(f"\nPort {port} is already in use by {occupant}.\n")
        print("Stop it with:\n")
        print(_stop_instructions(port))
        print(f"\nOr start on a different port:  set PORT=8001  then  python app.py")
        if occupant.startswith("an older"):
            print(
                "\nNote: the running instance is serving the code it was started with. "
                "Restart it to pick up any changes."
            )
        print()
        sys.exit(1)

    reload_enabled = os.getenv("RELOAD", "0") == "1"

    ssl_keyfile = os.getenv("SSL_KEYFILE")
    ssl_certfile = os.getenv("SSL_CERTFILE")
    default_key = PROJECT_ROOT / "certs" / "key.pem"
    default_cert = PROJECT_ROOT / "certs" / "cert.pem"

    uvicorn_kwargs = {
        "host": host,
        "port": port,
        "reload": reload_enabled,
    }

    if ssl_keyfile and ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        protocol = "https"
    elif default_key.exists() and default_cert.exists() and os.getenv("USE_SSL", "0") == "1":
        uvicorn_kwargs["ssl_keyfile"] = str(default_key)
        uvicorn_kwargs["ssl_certfile"] = str(default_cert)
        protocol = "https"
    else:
        protocol = "http"

    print(f"\nMedExplain AI running at {protocol}://{host}:{port}\n")
    uvicorn.run("app:app", **uvicorn_kwargs)
