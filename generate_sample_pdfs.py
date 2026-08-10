import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_sample_pdf(filename, title, patient_name, date, test_rows):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=12
    )

    story.append(Paragraph("<b>MEDEXPLAIN DIAGNOSTICS & LAB SERVICES</b>", title_style))
    story.append(Paragraph("108 Healthcare Blvd, Suite 400 | Patient Clinical Report", subtitle_style))
    story.append(Spacer(1, 10))

    # Patient Meta
    meta_data = [
        [Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']), Paragraph(f"<b>Report Date:</b> {date}", styles['Normal'])],
        [Paragraph(f"<b>Panel Type:</b> {title}", styles['Normal']), Paragraph("<b>Ordering Physician:</b> Dr. Sarah Jenkins, MD", styles['Normal'])],
        [Paragraph("<b>Status:</b> Completed", styles['Normal']), Paragraph("<b>Lab ID:</b> LAB-2026-8841", styles['Normal'])]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Table Header
    headers = ["Test Description", "Result Value", "Units", "Reference Interval", "Flag"]
    table_data = [[Paragraph(f"<b>{h}</b>", styles['Normal']) for h in headers]]

    for name, val, unit, ref, flag in test_rows:
        flag_color = colors.HexColor('#dc2626') if flag in ['HIGH', 'LOW'] else colors.HexColor('#16a34a')
        flag_p = Paragraph(f"<font color='{flag_color.hexval()}'><b>{flag}</b></font>", styles['Normal'])
        table_data.append([
            Paragraph(name, styles['Normal']),
            Paragraph(str(val), styles['Normal']),
            Paragraph(unit, styles['Normal']),
            Paragraph(ref, styles['Normal']),
            flag_p
        ])

    results_table = Table(table_data, colWidths=[160, 80, 80, 130, 90])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ALIGN', (1,0), (4,-1), 'CENTER'),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<font size=8 color='#94a3b8'>Notice: This lab report contains preliminary clinical data. Please consult your physician for definitive diagnosis.</font>", styles['Normal']))

    doc.build(story)
    print(f"Generated {filename}")

def main():
    os.makedirs("sample_reports", exist_ok=True)

    # CBC Report
    cbc_rows = [
        ("WBC (White Blood Cells)", "11.8", "x10^3/uL", "4.5 - 11.0", "HIGH"),
        ("RBC (Red Blood Cells)", "4.2", "x10^6/uL", "4.3 - 5.9", "LOW"),
        ("Hemoglobin", "11.5", "g/dL", "13.5 - 17.5", "LOW"),
        ("Hematocrit", "35.0", "%", "41.0 - 50.0", "LOW"),
        ("Platelets", "240", "x10^3/uL", "150 - 450", "NORMAL"),
    ]
    create_sample_pdf("sample_reports/sample_cbc_report.pdf", "Complete Blood Count (CBC)", "John Doe", "2026-08-01", cbc_rows)

    # Lipid Panel Report
    lipid_rows = [
        ("Total Cholesterol", "245", "mg/dL", "125 - 200", "HIGH"),
        ("HDL (Good Cholesterol)", "38", "mg/dL", "> 40", "LOW"),
        ("LDL (Bad Cholesterol)", "162", "mg/dL", "< 100", "HIGH"),
        ("Triglycerides", "210", "mg/dL", "< 150", "HIGH"),
    ]
    create_sample_pdf("sample_reports/sample_lipid_panel.pdf", "Lipid Panel", "Jane Smith", "2026-08-05", lipid_rows)

    # Comprehensive Metabolic Panel
    metabolic_rows = [
        ("Fasting Blood Glucose", "135", "mg/dL", "70 - 99", "HIGH"),
        ("BUN (Blood Urea Nitrogen)", "24", "mg/dL", "7 - 20", "HIGH"),
        ("Creatinine", "1.4", "mg/dL", "0.7 - 1.3", "HIGH"),
        ("Sodium", "139", "mEq/L", "136 - 145", "NORMAL"),
        ("Potassium", "4.8", "mEq/L", "3.5 - 5.1", "NORMAL"),
    ]
    create_sample_pdf("sample_reports/sample_metabolic_panel.pdf", "Comprehensive Metabolic Panel (CMP)", "Robert Johnson", "2026-08-08", metabolic_rows)

if __name__ == "__main__":
    main()
