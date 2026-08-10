"""
MedExplain AI — LLM Verification Script
Script: scripts/test_llm.py
Usage: python scripts/test_llm.py
"""

import sys
import os

# Add root project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.explanation.llm_client import GeminiClient, GeminiAPIError

def main() -> None:
    print("\n==========================================================================")
    print(" 🧪 MEDEXPLAIN AI — TESTING CORE LLM CLIENT (GEMINI API)")
    print("==========================================================================\n")

    test_input = "HbA1c: 8.2%. Fasting glucose: 145 mg/dL."
    prompt = f"""You are MedExplain AI, an educational healthcare assistant.
Explain the following medical laboratory values to a patient in simple, non-technical English.

HARD RULES:
- Do NOT provide a medical diagnosis or name a disease as a clinical conclusion.
- Do NOT recommend, name, or dose any medication.
- Always tell the patient to discuss these results with their doctor.

Patient Test Values:
{test_input}

Provide a brief, empathetic, patient-friendly explanation.
"""

    try:
        client = GeminiClient()
        response = client.generate(prompt=prompt, temperature=0.2)
        print("✅ SUCCESS: Gemini API Call Returned Response:\n")
        print("--------------------------------------------------------------------------")
        print(response)
        print("--------------------------------------------------------------------------\n")
    except GeminiAPIError as e:
        print(f"❌ GEMINI API ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
