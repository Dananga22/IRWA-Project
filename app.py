import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from extractor import process_pdf_report
from llm_agent import explain_lab_report
from demo_presentation import step3_llm_explanation_agent, step4_database_agent

app = FastAPI(
    title="MedExplain AI",
    description="Agentic AI Medical Lab Report Explanation System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/analysis", response_class=HTMLResponse)
async def serve_analysis():
    with open("static/analysis.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/v1/upload")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_uploads/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Agent 1 & 2: PDF Parsing + NLP Extraction
        extracted_data = process_pdf_report(temp_path)
        
        # Agent 3: LLM Explanation Agent
        annotated_items = step3_llm_explanation_agent(extracted_data)
        
        # Agent 4: Database Persistence Agent
        step4_database_agent(
            extracted_data["metadata"]["patient_name"],
            extracted_data["metadata"]["panel_type"],
            annotated_items
        )

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "metadata": extracted_data["metadata"],
            "total_tests": extracted_data["total_tests"],
            "abnormal_count": extracted_data["abnormal_count"],
            "test_items": annotated_items
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/v1/analyze-sample")
async def analyze_sample_report(sample_name: str):
    allowed_samples = {
        "cbc": "sample_reports/sample_cbc_report.pdf",
        "lipid": "sample_reports/sample_lipid_panel.pdf",
        "metabolic": "sample_reports/sample_metabolic_panel.pdf"
    }

    if sample_name not in allowed_samples:
        raise HTTPException(status_code=400, detail="Invalid sample selection.")

    pdf_path = allowed_samples[sample_name]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Sample PDF not found on server.")

    extracted_data = process_pdf_report(pdf_path)
    annotated_items = step3_llm_explanation_agent(extracted_data)

    step4_database_agent(
        extracted_data["metadata"]["patient_name"],
        extracted_data["metadata"]["panel_type"],
        annotated_items
    )

    return JSONResponse(content={
        "success": True,
        "filename": os.path.basename(pdf_path),
        "metadata": extracted_data["metadata"],
        "total_tests": extracted_data["total_tests"],
        "abnormal_count": extracted_data["abnormal_count"],
        "test_items": annotated_items
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
