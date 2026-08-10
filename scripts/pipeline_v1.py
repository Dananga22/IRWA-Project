"""
MedExplain AI — End-to-End Pipeline Proof of Concept (v1)
Script: scripts/pipeline_v1.py
Usage: python scripts/pipeline_v1.py data/samples/report1.pdf
"""

import sys
import os
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.document.processor import extract_text, DocumentProcessingError
from agents.extraction.extractor import extract_values
from agents.explanation.llm_client import GeminiClient, GeminiAPIError

SYSTEM_INSTRUCTION = """You are MedExplain AI, an empathetic and clear medical explanation assistant.
Your goal is to explain laboratory test results to a patient in simple, non-technical English.

HARD SAFETY RULES:
- Never provide a clinical diagnosis or name a specific disease as a definitive conclusion.
- Never recommend, name, or dose any medication or treatment.
- Never instruct a patient to alter or stop an existing medical treatment.
- ALWAYS instruct the patient to discuss these laboratory results with their primary physician.
"""

def run_pipeline(pdf_path: str) -> None:
    print("\n==========================================================================")
    print(" 🚀 MEDEXPLAIN AI — PIPELINE V1 EXECUTION (PROOF OF CONCEPT)")
    print("==========================================================================")
    print(f"Target PDF File: {pdf_path}\n")

    # STAGE 1: Document Processing Agent
    print("--------------------------------------------------------------------------")
    print(" 📄 STAGE 1: Document Processing Agent (PyMuPDF)")
    print("--------------------------------------------------------------------------")
    t0 = time.time()
    try:
        doc_text = extract_text(pdf_path)
        t_doc = time.time() - t0
        print(f"✅ Text extracted in {t_doc:.3f}s")
        print(f"   Page Count: {doc_text.page_count} | Scanned Flag: {doc_text.likely_scanned}")
        print(f"   Raw Text Snippet ({len(doc_text.raw_text)} chars):\n   {doc_text.raw_text[:200]}...\n")
    except DocumentProcessingError as e:
        print(f"❌ Document Processing Error: {e}")
        return

    # STAGE 2: NLP Extraction Agent
    print("--------------------------------------------------------------------------")
    print(" 🔬 STAGE 2: NLP Extraction Agent (Regex & Alias Normalization)")
    print("--------------------------------------------------------------------------")
    t1 = time.time()
    extracted_values = extract_values(doc_text.raw_text)
    t_nlp = time.time() - t1
    print(f"✅ Extracted {len(extracted_values)} lab values in {t_nlp:.3f}s:\n")

    if not extracted_values:
        print("⚠️ No laboratory values detected in document text.")
        return

    values_summary_lines = []
    for idx, v in enumerate(extracted_values, 1):
        line_str = f"  {idx}. {v.test_name}: {v.value} {v.unit} (Ref: {v.reference_range}) [{v.flag}]"
        print(line_str)
        values_summary_lines.append(f"- {v.test_name}: {v.value} {v.unit} (Normal range: {v.reference_range}) -> Result status: {v.flag}")

    # STAGE 3: LLM Explanation Agent
    print("\n--------------------------------------------------------------------------")
    print(" 🧠 STAGE 3: LLM Explanation Agent (Gemini API)")
    print("--------------------------------------------------------------------------")
    prompt = f"""{SYSTEM_INSTRUCTION}

Please explain the following extracted laboratory report biomarkers to the patient in plain English:

Biomarkers Evaluated:
{chr(10).join(values_summary_lines)}

Format your response cleanly with headings:
- 📊 Executive Summary
- 🔬 Result Breakdown & Plain Language Explanation
- 💡 What This Could Mean (Educational Context Only)
- ❓ Questions for Your Doctor
- ⚠️ Safety Disclaimer
"""
    t2 = time.time()
    try:
        client = GeminiClient()
        explanation = client.generate(prompt=prompt, temperature=0.2)
        t_llm = time.time() - t2
        print(f"✅ LLM Explanation generated in {t_llm:.3f}s\n")
        print("--------------------------------------------------------------------------")
        print(explanation)
        print("--------------------------------------------------------------------------")
    except GeminiAPIError as e:
        t_llm = time.time() - t2
        print(f"⚠️ LLM API Error after {t_llm:.3f}s: {e}")

    total_time = t_doc + t_nlp + t_llm
    print("\n==========================================================================")
    print(f" ⏱️ PIPELINE SUMMARY TIMINGS: Total: {total_time:.3f}s (Doc: {t_doc:.3f}s | NLP: {t_nlp:.3f}s | LLM: {t_llm:.3f}s)")
    print("==========================================================================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
    else:
        target_pdf = "data/samples/report1.pdf"

    run_pipeline(target_pdf)
