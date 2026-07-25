"""
Verification Report Service — Phase 8.

Orchestrates all analysis modules and generates a downloadable PDF report.

Pipeline:
    1. Run fake-news prediction
    2. Run claim extraction + per-claim prediction  (optional)
    3. Run evidence retrieval                       (optional)
    4. Run propaganda detection                     (optional)
    5. Run source credibility check                 (optional, requires URL)
    6. Assemble results → generate PDF via reportlab

Design:
    - reportlab is used for PDF generation (pure Python, no system deps).
    - Each section is a separate _draw_* method for clean separation.
    - Report is written to an in-memory BytesIO buffer (no disk I/O).
    - The buffer is returned to the caller for streaming.
"""

import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_RED    = (0.85, 0.15, 0.15)
_GREEN  = (0.10, 0.60, 0.20)
_ORANGE = (0.90, 0.50, 0.05)
_BLUE   = (0.10, 0.30, 0.70)
_GREY   = (0.45, 0.45, 0.45)
_BLACK  = (0.05, 0.05, 0.05)
_WHITE  = (1.00, 1.00, 1.00)
_LIGHT  = (0.95, 0.95, 0.97)


def generate_report(
    text: str,
    model: str = "logistic",
    source_url: Optional[str] = None,
    include_claims: bool = True,
    include_evidence: bool = True,
    include_propaganda: bool = True,
    include_credibility: bool = True,
    top_k_evidence: int = 3,
) -> BytesIO:
    """
    Run all requested analyses and generate a PDF report.

    Returns a BytesIO buffer containing the PDF binary.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas

    # ------------------------------------------------------------------
    # 1. Run analyses
    # ------------------------------------------------------------------
    logger.info("Report generation started — model=%s, text_len=%d", model, len(text))

    # Prediction
    from app.api.prediction import _run_model
    from app.services.explainability_service import (
        clickbait_score, extract_keywords, generate_explanation
    )
    prediction_result = _run_model(model, text)
    keywords = extract_keywords(text)
    cb_score = clickbait_score(text)
    explanation = generate_explanation(prediction_result["prediction"], cb_score)

    # Claims
    claims_result = None
    if include_claims:
        try:
            from app.services.claim_service import extract_claims, compute_overall_verdict
            claims = extract_claims(text)
            claim_verdicts = []
            preds, confs = [], []
            for c in claims[:10]:  # cap at 10 for report size
                r = _run_model(model, c)
                claim_verdicts.append({"text": c, **r})
                preds.append(r["prediction"])
                confs.append(r["confidence"])
            verdict, avg_conf = compute_overall_verdict(preds, confs)
            claims_result = {
                "claims": claim_verdicts,
                "overall_verdict": verdict,
                "overall_confidence": avg_conf,
            }
        except Exception as exc:
            logger.warning("Claims section skipped: %s", exc)

    # Evidence
    evidence_result = None
    if include_evidence:
        try:
            from app.services.retrieval_service import search_evidence
            evidence_result = search_evidence(text, top_k=top_k_evidence)
        except Exception as exc:
            logger.warning("Evidence section skipped: %s", exc)

    # Propaganda
    propaganda_result = None
    if include_propaganda:
        from app.services.propaganda_service import detect_propaganda
        propaganda_result = detect_propaganda(text)

    # Credibility
    credibility_result = None
    if include_credibility and source_url:
        try:
            from app.services.credibility_service import check_credibility
            credibility_result = check_credibility(source_url)
        except Exception as exc:
            logger.warning("Credibility section skipped: %s", exc)

    # ------------------------------------------------------------------
    # 2. Build PDF
    # ------------------------------------------------------------------
    buffer = BytesIO()
    page_w, page_h = A4
    c = rl_canvas.Canvas(buffer, pagesize=A4)

    margin = 20 * mm
    content_w = page_w - 2 * margin
    y = page_h - margin  # current vertical position (top-down)

    def new_page():
        nonlocal y
        c.showPage()
        y = page_h - margin

    def check_space(needed: float):
        if y - needed < margin + 10 * mm:
            new_page()

    # -- Header --
    c.setFillColorRGB(*_BLUE)
    c.rect(0, page_h - 30 * mm, page_w, 30 * mm, fill=True, stroke=False)
    c.setFillColorRGB(*_WHITE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, page_h - 18 * mm, "AI-Powered News Verification Report")
    c.setFont("Helvetica", 9)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    c.drawRightString(page_w - margin, page_h - 18 * mm, ts)
    y = page_h - 35 * mm

    # -- Metadata row --
    c.setFillColorRGB(*_LIGHT)
    c.rect(margin, y - 12 * mm, content_w, 12 * mm, fill=True, stroke=False)
    c.setFillColorRGB(*_BLACK)
    c.setFont("Helvetica", 9)
    meta_text = (
        f"Model: {model.upper()}    "
        f"Text length: {len(text)} chars    "
        f"Source: {source_url or 'Not provided'}"
    )
    c.drawString(margin + 3 * mm, y - 8 * mm, meta_text)
    y -= 16 * mm

    # -- Section: Prediction --
    y = _draw_section_header(c, "1. Fake News Prediction", y, margin, content_w, page_h, margin)
    pred = prediction_result["prediction"]
    conf = prediction_result["confidence"]
    colour = _RED if pred == "FAKE" else _GREEN
    c.setFont("Helvetica-Bold", 28)
    c.setFillColorRGB(*colour)
    c.drawString(margin, y - 15 * mm, pred)
    c.setFont("Helvetica", 11)
    c.setFillColorRGB(*_BLACK)
    c.drawString(margin + 30 * mm, y - 15 * mm, f"Confidence: {conf * 100:.1f}%")
    y -= 20 * mm

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(*_GREY)
    y = _draw_wrapped(c, explanation, margin, y, content_w, 10)
    y -= 4 * mm

    # Keywords
    if keywords:
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(*_BLACK)
        c.drawString(margin, y, "Keywords:")
        c.setFont("Helvetica", 10)
        c.drawString(margin + 25 * mm, y, "  |  ".join(keywords))
        y -= 5 * mm

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(*_GREY)
    c.drawString(margin, y, f"Clickbait score: {cb_score}/100")
    y -= 8 * mm

    # -- Section: Claims --
    if claims_result:
        check_space(40 * mm)
        y = _draw_section_header(c, "2. Claim-Level Analysis", y, margin, content_w, page_h, margin)
        ov = claims_result["overall_verdict"]
        ov_col = _RED if ov == "FAKE" else _GREEN
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(*ov_col)
        c.drawString(margin, y, f"Overall verdict: {ov}  ({claims_result['overall_confidence']*100:.1f}%)")
        y -= 6 * mm
        c.setFillColorRGB(*_BLACK)

        for i, cl in enumerate(claims_result["claims"], 1):
            check_space(14 * mm)
            bullet_col = _RED if cl["prediction"] == "FAKE" else _GREEN
            c.setFillColorRGB(*bullet_col)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(margin, y, f"[{cl['prediction']} {cl['confidence']*100:.0f}%]")
            c.setFillColorRGB(*_BLACK)
            c.setFont("Helvetica", 9)
            short = cl["text"][:120] + ("…" if len(cl["text"]) > 120 else "")
            c.drawString(margin + 28 * mm, y, short)
            y -= 5 * mm

        y -= 4 * mm

    # -- Section: Evidence --
    if evidence_result:
        check_space(40 * mm)
        y = _draw_section_header(c, "3. Similar Evidence Found", y, margin, content_w, page_h, margin)
        for ev in evidence_result:
            check_space(18 * mm)
            sim = ev.get("similarity", 0)
            label = ev.get("label", "")
            lc = _RED if label == "FAKE" else _GREEN
            c.setFillColorRGB(*lc)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(margin, y, f"[{label}  {sim*100:.0f}% match]")
            c.setFillColorRGB(*_BLACK)
            c.setFont("Helvetica-Bold", 9)
            title = ev.get("title", "")[:90]
            c.drawString(margin + 30 * mm, y, title)
            y -= 4 * mm
            c.setFont("Helvetica", 8)
            c.setFillColorRGB(*_GREY)
            snippet = ev.get("snippet", "")[:110]
            c.drawString(margin + 4 * mm, y, snippet)
            c.setFillColorRGB(*_BLACK)
            y -= 6 * mm
        y -= 2 * mm

    # -- Section: Propaganda --
    if propaganda_result:
        check_space(30 * mm)
        y = _draw_section_header(c, "4. Propaganda Analysis", y, margin, content_w, page_h, margin)
        ps = propaganda_result["overall_score"]
        ps_col = _RED if ps >= 0.7 else _ORANGE if ps >= 0.4 else _GREEN
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(*ps_col)
        c.drawString(margin, y, f"Propaganda Score: {ps*100:.0f}%")
        y -= 5 * mm
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(*_GREY)
        y = _draw_wrapped(c, propaganda_result["summary"], margin, y, content_w, 9)
        y -= 3 * mm

        for tech in propaganda_result.get("techniques_found", []):
            check_space(10 * mm)
            c.setFillColorRGB(*_BLACK)
            c.setFont("Helvetica-Bold", 9)
            c.drawString(margin, y, f"• {tech['technique']}  ({tech['confidence']*100:.0f}%)")
            c.setFont("Helvetica", 8)
            c.setFillColorRGB(*_GREY)
            phrases = ", ".join(tech.get("matched_phrases", [])[:3])
            if phrases:
                c.drawString(margin + 4 * mm, y - 4 * mm, f"  Matched: {phrases}")
                y -= 4 * mm
            y -= 5 * mm
        y -= 3 * mm

    # -- Section: Credibility --
    if credibility_result and credibility_result.get("found_in_database"):
        check_space(30 * mm)
        y = _draw_section_header(c, "5. Source Credibility", y, margin, content_w, page_h, margin)
        cl_label = credibility_result.get("credibility_label", "UNKNOWN")
        cl_col = (
            _GREEN if cl_label == "HIGH"
            else _ORANGE if cl_label == "MEDIUM"
            else _RED
        )
        c.setFont("Helvetica-Bold", 11)
        c.setFillColorRGB(*cl_col)
        c.drawString(margin, y, f"{credibility_result.get('domain', '')}  —  {cl_label}")
        y -= 5 * mm
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(*_BLACK)
        ts_val = credibility_result.get("trust_score")
        rs_val = credibility_result.get("reliability_score")
        bias = credibility_result.get("bias_rating", "")
        c.drawString(margin, y,
            f"Trust: {ts_val:.0f}/100   Reliability: {rs_val:.0f}/100   Bias: {bias}")
        y -= 5 * mm
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(*_GREY)
        y = _draw_wrapped(c, credibility_result.get("verdict", ""), margin, y, content_w, 9)
        y -= 4 * mm

    # -- Footer --
    c.setFillColorRGB(*_LIGHT)
    c.rect(0, 0, page_w, 12 * mm, fill=True, stroke=False)
    c.setFillColorRGB(*_GREY)
    c.setFont("Helvetica", 8)
    c.drawCentredString(page_w / 2, 4 * mm,
        "Generated by AI-Powered News Verification Platform  |  For informational purposes only")

    c.save()
    buffer.seek(0)
    logger.info("Report PDF generated — %d bytes", buffer.getbuffer().nbytes)
    return buffer


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_section_header(
    c, title: str, y: float,
    margin: float, content_w: float,
    page_h: float, page_margin: float
) -> float:
    from reportlab.lib.units import mm
    if y - 14 * mm < page_margin + 10 * mm:
        c.showPage()
        y = page_h - page_margin

    c.setFillColorRGB(*_BLUE)
    c.rect(margin, y - 10 * mm, content_w, 10 * mm, fill=True, stroke=False)
    c.setFillColorRGB(*_WHITE)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin + 3 * mm, y - 7 * mm, title)
    return y - 14 * mm


def _draw_wrapped(
    c, text: str, x: float, y: float,
    max_width: float, font_size: int = 9,
    line_height: float = None
) -> float:
    from reportlab.lib.units import mm
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if line_height is None:
        line_height = font_size * 1.4

    c.setFont("Helvetica", font_size)
    words = text.split()
    line = ""

    for word in words:
        test = f"{line} {word}".strip()
        if stringWidth(test, "Helvetica", font_size) <= max_width:
            line = test
        else:
            if line:
                c.drawString(x, y, line)
                y -= line_height
            line = word

    if line:
        c.drawString(x, y, line)
        y -= line_height

    return y
