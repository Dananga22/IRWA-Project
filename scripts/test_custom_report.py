import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_sample_pdfs import create_sample_pdf
from medexplain.graph import run_pipeline


def test_custom():
    os.makedirs("sample_reports", exist_ok=True)
    custom_pdf = "sample_reports/test_custom_report.pdf"
    
    # Create custom report with 5 items
    rows = [
        ("Hemoglobin", "11.2", "g/dL", "13.5 - 17.5", "LOW"),
        ("Fasting Blood Sugar", "115", "mg/dL", "70 - 99", "HIGH"),
        ("Serum Creatinine", "1.45", "mg/dL", "0.7 - 1.3", "HIGH"),
        ("Platelets", "210", "x10^3/uL", "150 - 450", "NORMAL"),
        ("Total Cholesterol", "210", "mg/dL", "125 - 200", "HIGH"),
    ]
    create_sample_pdf(custom_pdf, "Custom Lab Report", "Alice Smith", "2026-08-12", rows)
    
    print("\n--- Running pipeline on custom PDF report ---")
    state = run_pipeline(custom_pdf, trace_id="test_custom_trace")
    
    print("\nPipeline execution complete:")
    print("Status:", state.get("status"))
    print("Total Tests Extracted:", state.get("total_tests"))
    print("Approved:", state.get("approved"))
    print("Test Items:")
    for item in state.get("test_items", []):
        print(f"  - {item.get('test_name')}: {item.get('value')} {item.get('unit')} (ref {item.get('reference_range')}) [{item.get('flag')}]")

if __name__ == "__main__":
    test_custom()
