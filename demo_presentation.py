"""
================================================================================
MedExplain AI — Live Mid-Evaluation Progress Demo
Architecture: Document Processing Agent -> NLP Extraction Agent -> LLM Explanation Agent -> DB Agent
================================================================================
"""

import os
import sys
import re
import json
import time
import sqlite3
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 0: Dependency Checks & Fallback Setup
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF not found. Please install using: pip install pymupdf")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import mysql.connector
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

# ==============================================================================
# STEP 1: DOCUMENT PROCESSING AGENT
# ==============================================================================
def step1_document_processing_agent(pdf_path: str) -> str:
    """
    [AGENT 1: Document Processing Agent]
    Task: Ingest PDF lab report and extract raw text content using PyMuPDF.
    """
    print("\n" + "=" * 80)
    print(" 📄 [AGENT 1: DOCUMENT PROCESSING AGENT]")
    print(f"    Target File: {pdf_path}")
    print("=" * 80)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    print("   [1.1] Opening PDF document via PyMuPDF (fitz)...")
    doc = fitz.open(pdf_path)
    print(f"   [1.2] Document opened successfully. Total Pages: {len(doc)}")

    raw_text_chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        raw_text_chunks.append(text)
        print(f"   [1.3] Page {i+1} extracted ({len(text)} characters).")

    full_text = "\n".join(raw_text_chunks)
    print("\n   🔍 RAW TEXT PREVIEW (First 250 characters):")
    print("   " + "-" * 70)
    print("   " + full_text.strip().replace("\n", " ")[:250] + "...")
    print("   " + "-" * 70)

    return full_text


# ==============================================================================
# STEP 2: NLP EXTRACTION AGENT
# ==============================================================================
def step2_nlp_extraction_agent(pdf_path: str, raw_text: str) -> Dict[str, Any]:
    """
    [AGENT 2: NLP Extraction Agent]
    Task: Extract patient metadata and structured test key-value pairs using Regex & PyMuPDF tables.
    """
    print("\n" + "=" * 80)
    print(" 🔬 [AGENT 2: NLP EXTRACTION AGENT]")
    print("    Task: Parsing Patient Meta & Extracting Lab Biomarkers (Regex + Tabular NLP)")
    print("=" * 80)

    # 2.1 Metadata Regex Parsing
    metadata = {
        "patient_name": "John Doe",
        "report_date": "2026-08-10",
        "panel_type": "Medical Lab Panel"
    }

    clean_str = " | ".join([line.strip() for line in raw_text.split("\n") if line.strip()])
    
    name_match = re.search(r"Patient Name:\s*([^|]+)", clean_str, re.IGNORECASE)
    if name_match:
        metadata["patient_name"] = name_match.group(1).strip()

    date_match = re.search(r"Report Date:\s*([^|]+)", clean_str, re.IGNORECASE)
    if date_match:
        metadata["report_date"] = date_match.group(1).strip()

    panel_match = re.search(r"Panel Type:\s*([^|]+)", clean_str, re.IGNORECASE)
    if panel_match:
        metadata["panel_type"] = panel_match.group(1).strip()

    print(f"   [2.1] Extracted Patient: {metadata['patient_name']} | Date: {metadata['report_date']}")

    # 2.2 Table & Regex Lab Items Extraction
    test_items = []
    doc = fitz.open(pdf_path)

    # Method A: PyMuPDF Table Structure Parsing
    for page in doc:
        tabs = page.find_tables()
        for tab in tabs:
            for row in tab.extract():
                if not row or len(row) < 4:
                    continue
                clean_row = [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
                if "Test Description" in clean_row[0] or "Result" in clean_row[0]:
                    continue
                try:
                    name = clean_row[0]
                    val = float(clean_row[1])
                    unit = clean_row[2] if len(clean_row) > 2 else ""
                    ref = clean_row[3] if len(clean_row) > 3 else ""
                    flag = clean_row[4].upper() if len(clean_row) > 4 and clean_row[4] else "NORMAL"
                    
                    test_items.append({
                        "test_name": name,
                        "value": val,
                        "unit": unit,
                        "reference_range": ref,
                        "flag": flag
                    })
                except ValueError:
                    continue

    # Method B: Regex Pattern Fallback (e.g. "HbA1c: 8.2%" or standard text format)
    if not test_items:
        regex_patterns = [
            # Pattern: Name: Value Unit (e.g. HbA1c: 8.2 %)
            r"(?P<name>[A-Za-z0-9\s\(\)\-\/]+?):\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>[%a-zA-Z0-9\/\^]+)",
            # Pattern: Name Value Unit Range Flag
            r"(?P<name>[A-Za-z0-9\s\(\)\-\/]+?)\s+(?P<val>\d+(?:\.\d+)?)\s+(?P<unit>[%a-zA-Z0-9\/\^]+)\s+(?P<ref>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?)\s+(?P<flag>HIGH|LOW|NORMAL)"
        ]
        for line in raw_text.split("\n"):
            line_s = line.strip()
            for pat in regex_patterns:
                m = re.search(pat, line_s, re.IGNORECASE)
                if m:
                    test_items.append({
                        "test_name": m.group("name").strip(),
                        "value": float(m.group("val")),
                        "unit": m.group("unit").strip(),
                        "reference_range": m.group("ref").strip() if "ref" in m.groupdict() and m.group("ref") else "N/A",
                        "flag": m.group("flag").upper() if "flag" in m.groupdict() and m.group("flag") else "EVALUATE"
                    })
                    break

    print(f"   [2.2] Successfully Extracted {len(test_items)} Biomarker Items:")
    for item in test_items:
        print(f"       • {item['test_name']:<28} = {item['value']} {item['unit']} (Ref: {item['reference_range']}) [{item['flag']}]")

    return {
        "metadata": metadata,
        "test_items": test_items
    }


# ==============================================================================
# STEP 3: LLM EXPLANATION AGENT (Gemini API + Fallback)
# ==============================================================================
SYSTEM_SAFETY_PROMPT = """You are MedExplain AI, an educational healthcare explanation assistant.
Your task is to explain lab test values to patients in friendly, plain English.

STRICT CONSTRAINTS & RESPONSIBLE AI RULES:
1. You are NOT a doctor and must NEVER provide a medical diagnosis or prescribe treatment.
2. Explain what the test value generally means in simple, non-jargon terms.
3. State whether the value appears normal, high, or low based on the provided reference range.
4. Keep the explanation concise (2-3 sentences per test).
5. Always remind the patient to consult their physician for clinical diagnosis.
"""

def generate_mock_item_explanation(item: dict) -> str:
    """Bulletproof local fallback explanation generator."""
    name = item['test_name']
    val = item['value']
    unit = item['unit']
    flag = item['flag']
    ref = item['reference_range']

    if flag == "HIGH":
        return f"{name} level is {val} {unit}, which is above the standard reference range of {ref}. A high result can occur for various reasons including diet, stress, or mild inflammation. This is for educational purposes only—please consult your physician for clinical interpretation."
    elif flag == "LOW":
        return f"{name} level is {val} {unit}, which is below the normal reference range of {ref}. Low levels may indicate nutritional factors or physiological variation. Note: This is an educational summary, not a medical diagnosis. Please discuss this with your doctor."
    else:
        return f"{name} level is {val} {unit}, which falls within the typical reference range of {ref}. This indicates normal baseline levels. Always review your complete results with your healthcare provider."

def step3_llm_explanation_agent(extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    [AGENT 3: LLM Explanation Agent]
    Task: Generate short, empathetic, non-diagnostic plain-English explanations using Gemini API.
    Includes rate-limit & network error fallback mechanism.
    """
    print("\n" + "=" * 80)
    print(" 🤖 [AGENT 3: LLM EXPLANATION AGENT (Gemini API)]")
    print("    Task: Generating Patient-Friendly Explanations with Responsible AI Safety Prompts")
    print("=" * 80)

    api_key = os.getenv("GEMINI_API_KEY")
    test_items = extracted_data["test_items"]
    annotated_items = []

    client = None
    if HAS_GENAI and api_key and api_key != "your_gemini_api_key_here":
        try:
            client = genai.Client(api_key=api_key)
            print("   [3.1] Initialized Gemini API Client (`gemini-2.5-flash`).")
        except Exception as e:
            print(f"   ⚠️ Gemini Client init notice: {e}. Using fallback generator.")

    for i, item in enumerate(test_items):
        print(f"\n   [3.2] Processing Item {i+1}/{len(test_items)}: {item['test_name']}...")
        
        explanation = ""
        if client:
            try:
                user_prompt = f"""Explain this lab test result for a patient:
Test Name: {item['test_name']}
Measured Value: {item['value']} {item['unit']}
Reference Range: {item['reference_range']}
Flag Status: {item['flag']}

Provide a 2-3 sentence friendly explanation. Remind them to check with their physician."""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_SAFETY_PROMPT,
                        temperature=0.2
                    )
                )
                explanation = response.text.strip()
                print(f"       ✅ Live Gemini API Explanation Generated.")
            except Exception as err:
                print(f"       ⚠️ Gemini API Call warning ({err}). Using fallback generator.")
                explanation = generate_mock_item_explanation(item)
        else:
            print("       ℹ️ Running in Local Safety Mock Mode (No Gemini API Key set).")
            explanation = generate_mock_item_explanation(item)

        print(f"       💬 Explanation preview:\n          \"{explanation[:120]}...\"")

        item_copy = dict(item)
        item_copy["explanation"] = explanation
        annotated_items.append(item_copy)
        time.sleep(0.3)  # Rate limit protection for live demo

    return annotated_items


# ==============================================================================
# STEP 4: DATABASE PERSISTENCE AGENT (MySQL + SQLite Fallback)
# ==============================================================================
def step4_database_agent(patient_name: str, panel_type: str, annotated_items: List[Dict[str, Any]]):
    """
    [AGENT 4: Database Persistence Agent]
    Task: Save report records to MySQL database `reports` table.
    Includes SQLite auto-fallback so presentation NEVER crashes if MySQL is offline.
    """
    print("\n" + "=" * 80)
    print(" 💾 [AGENT 4: DATABASE PERSISTENCE AGENT]")
    print("    Task: Persisting Extracted & Explained Data into SQL Database")
    print("=" * 80)

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", 3306))
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "medexplain_db")

    conn = None
    use_sqlite = False

    # Attempt MySQL connection
    if HAS_MYSQL:
        try:
            print(f"   [4.1] Connecting to MySQL server ({db_host}:{db_port}/{db_name})...")
            conn = mysql.connector.connect(
                host=db_host,
                port=db_port,
                user=db_user,
                password=db_password,
                database=db_name
            )
            print("   ✅ Connected to MySQL database successfully!")
        except Exception as db_err:
            print(f"   ⚠️ MySQL Connection Notice ({db_err}).")
            print("   🛡️ Activating Live Demo Protection: Switching to SQLite Database (`medexplain_demo.db`).")
            use_sqlite = True
    else:
        print("   ℹ️ MySQL driver not installed. Using SQLite Database (`medexplain_demo.db`).")
        use_sqlite = True

    if use_sqlite:
        conn = sqlite3.connect("medexplain_demo.db")

    cursor = conn.cursor()

    # Create Table if Not Exists
    if use_sqlite:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                panel_type TEXT,
                test_name TEXT,
                test_value REAL,
                unit TEXT,
                reference_range TEXT,
                flag TEXT,
                explanation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    else:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_name VARCHAR(100) NOT NULL,
                panel_type VARCHAR(100),
                test_name VARCHAR(100) NOT NULL,
                test_value DOUBLE NOT NULL,
                unit VARCHAR(50),
                reference_range VARCHAR(50),
                flag VARCHAR(20),
                explanation TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    print(f"   [4.2] Table `reports` verified. Inserting {len(annotated_items)} records...")

    insert_sql = """
        INSERT INTO reports (patient_name, panel_type, test_name, test_value, unit, reference_range, flag, explanation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """ if not use_sqlite else """
        INSERT INTO reports (patient_name, panel_type, test_name, test_value, unit, reference_range, flag, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    for item in annotated_items:
        params = (
            patient_name,
            panel_type,
            item['test_name'],
            item['value'],
            item['unit'],
            item['reference_range'],
            item['flag'],
            item['explanation']
        )
        cursor.execute(insert_sql, params)

    conn.commit()
    print("   ✅ All records committed to database!")

    # Read back saved rows to prove persistence
    print("\n   [4.3] READING BACK PERSISTED ROWS FROM DATABASE:")
    print("   " + "-" * 75)
    select_sql = "SELECT id, patient_name, test_name, test_value, unit, flag, created_at FROM reports ORDER BY id DESC LIMIT 5"
    cursor.execute(select_sql)
    rows = cursor.fetchall()

    for row in rows:
        print(f"   Row ID #{row[0]} | Patient: {row[1]:<12} | Test: {row[2]:<22} | Val: {row[3]} {row[4]} [{row[5]}]")

    print("   " + "-" * 75)
    cursor.close()
    conn.close()


# ==============================================================================
# MAIN LIVE DEMO EXECUTION PIPELINE
# ==============================================================================
def main():
    print("""
    ==========================================================================
    🩺 MEDEXPLAIN AI — MID-EVALUATION LIVE DEMO SYSTEM
    Architecture: PDF Ingestion -> Regex NLP -> Gemini LLM -> SQL DB
    ==========================================================================
    """)

    sample_pdf = "sample_reports/sample_cbc_report.pdf"
    if len(sys.argv) > 1:
        sample_pdf = sys.argv[1]

    # Step 1: Document Processing
    raw_text = step1_document_processing_agent(sample_pdf)

    # Step 2: NLP Extraction
    extracted_data = step2_nlp_extraction_agent(sample_pdf, raw_text)

    # Step 3: LLM Explanation
    annotated_items = step3_llm_explanation_agent(extracted_data)

    # Step 4: Database Persistence
    step4_database_agent(
        extracted_data["metadata"]["patient_name"],
        extracted_data["metadata"]["panel_type"],
        annotated_items
    )

    print("""
    ==========================================================================
    🎉 LIVE DEMO PIPELINE RUN COMPLETE!
    All multi-agent stages executed and data successfully persisted.
    ==========================================================================
    """)

if __name__ == "__main__":
    main()
