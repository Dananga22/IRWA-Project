import sys
import os
import json
import argparse
from extractor import process_pdf_report
from llm_agent import explain_lab_report

def run_pipeline_for_file(pdf_path: str, save_output: bool = True):
    print("=" * 80)
    print(f" 🩺 MEDEXPLAIN AI — PROCESSING LAB REPORT: {pdf_path}")
    print("=" * 80)

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File '{pdf_path}' not found.")
        return None

    # Step 1 & 2: PyMuPDF Extraction + NLP Parsing
    print("1️⃣ [PDF & NLP Extraction] Extracting text & parsing biomarkers...")
    extracted_data = process_pdf_report(pdf_path)

    meta = extracted_data["metadata"]
    print(f"   👤 Patient: {meta.get('patient_name')} | Date: {meta.get('report_date')}")
    print(f"   📋 Panel Type: {meta.get('panel_type')}")
    print(f"   📊 Total Biomarkers Parsed: {extracted_data['total_tests']} | Abnormal (High/Low): {extracted_data['abnormal_count']}")

    print("\n   [Extracted Biomarker Items]")
    for item in extracted_data["test_items"]:
        flag_str = f"[{item['flag']}]" if item['flag'] in ['HIGH', 'LOW'] else "[NORMAL]"
        print(f"   • {item['test_name']:<30} {item['value']:<8} {item['unit']:<10} Range: {item['reference_range']:<15} {flag_str}")

    # Step 3: LLM Explanation Agent
    print("\n2️⃣ [LLM Agent] Invoking MedExplain AI Explanation Agent...")
    explanation_md = explain_lab_report(extracted_data)

    print("\n" + "=" * 80)
    print(" 📖 GENERATED PATIENT EXPLANATION REPORT")
    print("=" * 80)
    print(explanation_md)
    print("=" * 80)

    # Save output artifacts
    if save_output:
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        json_path = f"output/{base_name}_extracted.json"
        md_path = f"output/{base_name}_explanation.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# MedExplain AI Explanation Report for {meta.get('patient_name')}\n\n")
            f.write(explanation_md)

        print(f"\n💾 Saved structured extraction JSON to: {json_path}")
        print(f"💾 Saved patient explanation Markdown to: {md_path}")

    return {
        "extracted_data": extracted_data,
        "explanation": explanation_md
    }

def main():
    parser = argparse.ArgumentParser(description="MedExplain AI - Phase 1 End-to-End Pipeline CLI")
    parser.add_argument("--pdf", type=str, help="Path to specific lab report PDF file to process")
    parser.add_argument("--all", action="store_true", help="Process all sample PDFs in sample_reports/")
    args = parser.parse_args()

    if args.pdf:
        run_pipeline_for_file(args.pdf)
    elif args.all or len(sys.argv) == 1:
        sample_files = [
            "sample_reports/sample_cbc_report.pdf",
            "sample_reports/sample_lipid_panel.pdf",
            "sample_reports/sample_metabolic_panel.pdf"
        ]
        print(f"🚀 Running MedExplain AI Pipeline across {len(sample_files)} sample report formats...\n")
        results = []
        for file in sample_files:
            if os.path.exists(file):
                res = run_pipeline_for_file(file)
                results.append(res)
            else:
                print(f"Skipping missing sample: {file}")

        print("\n" + "🎉" * 20)
        print(" MedExplain AI — Phase 1 Proof-of-Concept Pipeline Run Completed Successfully!")
        print(" " + "🎉" * 20)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
