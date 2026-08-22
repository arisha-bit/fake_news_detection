from io import BytesIO
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def build_multimodal_report(payload: dict) -> BytesIO:
    buffer = BytesIO()
    page_w, page_h = A4
    c = canvas.Canvas(buffer, pagesize=A4)
    margin = 18 * mm
    y = page_h - margin

    c.setFillColorRGB(0.12, 0.18, 0.43)
    c.rect(0, page_h - 24 * mm, page_w, 24 * mm, fill=True, stroke=False)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, page_h - 15 * mm, "Multimodal News Verification Report")
    c.setFont("Helvetica", 9)
    c.drawRightString(page_w - margin, page_h - 15 * mm, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    y -= 24 * mm
    c.setFillColorRGB(0.95, 0.97, 0.99)
    c.rect(margin, y - 8 * mm, page_w - 2 * margin, 8 * mm, fill=True, stroke=False)
    c.setFillColorRGB(0.15, 0.15, 0.15)
    c.setFont("Helvetica", 9)
    c.drawString(margin + 3 * mm, y - 5 * mm, f"Verdict: {payload.get('overall_verdict', 'UNCERTAIN')}")
    c.drawRightString(page_w - margin - 3 * mm, y - 5 * mm, f"Confidence: {payload.get('confidence', 0):.0%}")
    y -= 16 * mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Summary")
    y -= 7 * mm
    c.setFont("Helvetica", 10)
    text = payload.get("summary", "No summary available")
    for line in text.splitlines() or [text]:
        c.drawString(margin, y, line[:100])
        y -= 5 * mm

    c.save()
    buffer.seek(0)
    return buffer
