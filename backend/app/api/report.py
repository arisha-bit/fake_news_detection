"""
Verification Report API — POST /report/generate

Accepts analysis parameters, runs all requested modules, and returns
a downloadable PDF verification report as a StreamingResponse.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.report import ReportRequest
from app.services.report_service import generate_report

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/report",
    tags=["Verification Report"],
)


@router.post("/generate")
def generate_verification_report(
    payload: ReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a downloadable PDF verification report.

    Runs the requested analysis modules (prediction, claims, evidence,
    propaganda, source credibility) and compiles results into a
    professionally formatted PDF.

    - **text**: Article text to analyse.
    - **model**: `logistic` (default), `lstm`, or `bert`.
    - **source_url**: Optional — enables source credibility section.
    - **include_claims/evidence/propaganda/credibility**: Toggle sections.
    - **top_k_evidence**: Number of evidence items (default 3).

    Returns `application/pdf` as a streaming download.
    """
    logger.info(
        "Report requested — user=%s, model=%s, text_len=%d",
        current_user.id,
        payload.model,
        len(payload.text),
    )

    pdf_buffer = generate_report(
        text=payload.text,
        model=payload.model,
        source_url=payload.source_url,
        include_claims=payload.include_claims,
        include_evidence=payload.include_evidence,
        include_propaganda=payload.include_propaganda,
        include_credibility=payload.include_credibility,
        top_k_evidence=payload.top_k_evidence,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"verification_report_{timestamp}.pdf"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
