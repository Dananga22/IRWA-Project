import os
import json
from dotenv import load_dotenv

load_dotenv()

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

SYSTEM_PROMPT = """You are MedExplain AI, an intelligent and empathetic medical explanation agent.
Your primary role is to explain medical laboratory report results to patients in clear, accessible language.

Core Responsibilities:
1. Translate technical medical lab terminology into simple, reassuring plain language.
2. Clearly distinguish between NORMAL and ABNORMAL (HIGH/LOW) results.
3. For abnormal results, explain in layman terms what the biomarker does in the body and potential common reasons for variation.
4. DO NOT provide a definitive diagnosis or prescribe treatments.
5. Suggest 3-4 proactive, constructive questions the patient can ask their physician during their next visit.
6. Always end with a clear Responsible AI safety disclaimer.

Format your response in clean Markdown with clear headings:
- 📊 Executive Summary
- 🔬 Result Breakdown & Plain Language Explanation
- 💡 What This Could Mean (Educational Context Only)
- ❓ Questions for Your Doctor
- ⚠️ Medical Safety Disclaimer
"""

def generate_mock_explanation(report_data: dict) -> str:
    """Fallback generator when GEMINI_API_KEY is not available."""
    meta = report_data.get("metadata", {})
    patient = meta.get("patient_name", "Patient")
    panel = meta.get("panel_type", "Lab Report")
    abnormal = [i for i in report_data.get("test_items", []) if i["flag"] in ["HIGH", "LOW"]]
    normal = [i for i in report_data.get("test_items", []) if i["flag"] == "NORMAL"]

    output = f"""### 📊 Executive Summary
Hello **{patient}**, this is an automated plain-language summary of your **{panel}** report dated **{meta.get('report_date', 'N/A')}**.
Out of **{report_data.get('total_tests', 0)}** tests evaluated, **{len(normal)}** are within normal reference ranges and **{len(abnormal)}** require attention from your physician.

### 🔬 Result Breakdown & Plain Language Explanation
"""
    if abnormal:
        output += "#### ⚠️ Attention Needed (Outside Reference Range):\n"
        for item in abnormal:
            direction = "above" if item["flag"] == "HIGH" else "below"
            output += f"- **{item['test_name']}**: Measured at **{item['value']} {item['unit']}** ({item['flag']}). Normal range is `{item['reference_range']}`. This is **{direction}** typical baseline levels.\n"

    if normal:
        output += "\n#### ✅ Within Normal Limits:\n"
        for item in normal:
            output += f"- **{item['test_name']}**: **{item['value']} {item['unit']}** (`{item['reference_range']}`)\n"

    output += f"""
### 💡 What This Could Mean (Educational Context Only)
- **Understanding Abnormalities**: Test values fluctuate based on hydration, diet, exercise, stress, or minor infections. An out-of-range value does not necessarily mean illness.

### ❓ Questions for Your Doctor
1. "How do these results compare to my previous lab records?"
2. "Are any lifestyle modifications (such as dietary adjustments) recommended?"
3. "Should any of these tests be re-checked in 3 to 6 months?"

---
### ⚠️ Medical Safety Disclaimer
*MedExplain AI provides educational explanations of laboratory data for informational purposes only. It is **not** a diagnostic tool and does **not** replace professional advice, diagnosis, or treatment from a qualified healthcare provider.*
"""
    return output

def explain_lab_report(report_data: dict) -> str:
    """Generate LLM explanation for lab report data using Gemini API or mock fallback."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not HAS_GENAI or not api_key:
        print("[MedExplain AI] GEMINI_API_KEY not set or SDK missing. Running in Mock Explanation Mode.")
        return generate_mock_explanation(report_data)

    try:
        client = genai.Client(api_key=api_key)
        prompt_content = f"""Please explain the following extracted laboratory report data to the patient:

Patient Metadata: {json.dumps(report_data.get('metadata', {}))}
Total Biomarkers Evaluated: {report_data.get('total_tests', 0)}
Abnormal Count: {report_data.get('abnormal_count', 0)}

Test Results Data:
{json.dumps(report_data.get('test_items', []), indent=2)}

Please provide a compassionate, structured Markdown explanation following your system instructions.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            )
        )
        return response.text

    except Exception as e:
        print(f"[MedExplain AI] Gemini API call error: {e}. Falling back to mock generator.")
        return generate_mock_explanation(report_data)

if __name__ == "__main__":
    sample_data = {
        "metadata": {"patient_name": "Test User", "panel_type": "Complete Blood Count", "report_date": "2026-08-10"},
        "total_tests": 3,
        "abnormal_count": 2,
        "test_items": [
            {"test_name": "Hemoglobin", "value": 11.2, "unit": "g/dL", "reference_range": "13.5 - 17.5", "flag": "LOW"},
            {"test_name": "WBC", "value": 12.5, "unit": "x10^3/uL", "reference_range": "4.5 - 11.0", "flag": "HIGH"},
            {"test_name": "Platelets", "value": 250, "unit": "x10^3/uL", "reference_range": "150 - 450", "flag": "NORMAL"}
        ]
    }
    print("--- Testing llm_agent.py ---")
    result = explain_lab_report(sample_data)
    print(result)
