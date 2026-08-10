import fitz # PyMuPDF
import re

def extract_raw_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text_content = []
    for page in doc:
        text_content.append(page.get_text())
    return "\n".join(text_content)

def parse_metadata(text: str) -> dict:
    """Extract patient and report metadata using regular expressions."""
    metadata = {
        "patient_name": "Unknown Patient",
        "report_date": "Unknown Date",
        "panel_type": "Medical Lab Report",
        "ordering_physician": "Not Specified",
        "lab_id": "N/A"
    }

    # Clean text lines for header metadata regex
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_str = " | ".join(lines)

    name_match = re.search(r"Patient Name:\s*([^|]+)", full_str, re.IGNORECASE)
    if name_match:
        metadata["patient_name"] = name_match.group(1).strip()

    date_match = re.search(r"Report Date:\s*([^|]+)", full_str, re.IGNORECASE)
    if date_match:
        metadata["report_date"] = date_match.group(1).strip()

    panel_match = re.search(r"Panel Type:\s*([^|]+)", full_str, re.IGNORECASE)
    if panel_match:
        metadata["panel_type"] = panel_match.group(1).strip()

    physician_match = re.search(r"Ordering Physician:\s*([^|]+)", full_str, re.IGNORECASE)
    if physician_match:
        metadata["ordering_physician"] = physician_match.group(1).strip()

    lab_match = re.search(r"Lab ID:\s*([^|]+)", full_str, re.IGNORECASE)
    if lab_match:
        metadata["lab_id"] = lab_match.group(1).strip()

    return metadata

def parse_lab_test_items(pdf_path: str, raw_text: str) -> list[dict]:
    """
    Extract test result rows from PDF using PyMuPDF table extraction with Regex fallback.
    """
    lab_items = []
    doc = fitz.open(pdf_path)

    # Strategy 1: PyMuPDF find_tables (best for formatted table PDFs)
    for page in doc:
        tabs = page.find_tables()
        for tab in tabs:
            rows = tab.extract()
            for row in rows:
                if not row or len(row) < 4:
                    continue
                # Clean elements
                row_clean = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
                
                # Check if header row
                if "Test Description" in row_clean[0] or "Result" in row_clean[0]:
                    continue
                
                # Identify columns: Name, Value, Unit, Ref Interval, Flag
                # Standard format: [Name, Val, Unit, Ref, Flag]
                test_name = row_clean[0]
                val_str = row_clean[1] if len(row_clean) > 1 else ""
                unit = row_clean[2] if len(row_clean) > 2 else ""
                ref_range = row_clean[3] if len(row_clean) > 3 else ""
                flag = row_clean[4].upper() if len(row_clean) > 4 and row_clean[4] else "NORMAL"

                try:
                    val = float(val_str)
                    if flag not in ["HIGH", "LOW", "NORMAL"]:
                        flag = "NORMAL"
                    lab_items.append({
                        "test_name": test_name,
                        "value": val,
                        "unit": unit,
                        "reference_range": ref_range,
                        "flag": flag
                    })
                except ValueError:
                    continue

    # Strategy 2: Fallback Regex parsing if table extraction found no items
    if not lab_items:
        pattern = re.compile(
            r"(?P<name>[A-Za-z0-9\s\(\)\-\/\,]+?)\s+"
            r"(?P<val>\d+(?:\.\d+)?)\s+"
            r"(?P<unit>[a-zA-Z0-9\%\^\/\_]+)\s+"
            r"(?P<ref>(?:[\<\>]?\s*\d+(?:\.\d+)?\s*\-\s*\d+(?:\.\d+)?)|(?:[\<\>]\s*\d+(?:\.\d+)?))\s+"
            r"(?P<flag>HIGH|LOW|NORMAL)",
            re.IGNORECASE
        )
        for match in pattern.finditer(raw_text):
            lab_items.append({
                "test_name": match.group("name").strip(),
                "value": float(match.group("val")),
                "unit": match.group("unit").strip(),
                "reference_range": match.group("ref").strip(),
                "flag": match.group("flag").upper()
            })

    return lab_items

def process_pdf_report(pdf_path: str) -> dict:
    """Complete extraction pipeline for a single lab report PDF."""
    raw_text = extract_raw_text_from_pdf(pdf_path)
    metadata = parse_metadata(raw_text)
    test_items = parse_lab_test_items(pdf_path, raw_text)

    abnormal_count = sum(1 for item in test_items if item["flag"] in ["HIGH", "LOW"])

    return {
        "pdf_path": pdf_path,
        "metadata": metadata,
        "test_items": test_items,
        "total_tests": len(test_items),
        "abnormal_count": abnormal_count,
        "raw_text": raw_text
    }

if __name__ == "__main__":
    import json
    for sample in ["sample_reports/sample_cbc_report.pdf", "sample_reports/sample_lipid_panel.pdf", "sample_reports/sample_metabolic_panel.pdf"]:
        print(f"\n--- Testing extraction on {sample} ---")
        result = process_pdf_report(sample)
        print(json.dumps({
            "metadata": result["metadata"],
            "total_tests": result["total_tests"],
            "abnormal_count": result["abnormal_count"],
            "items": result["test_items"]
        }, indent=2))
