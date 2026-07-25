"""
Source Credibility API — POST /credibility/check

Accepts a URL or domain and returns a structured credibility assessment
including trust score, bias rating, reliability score, and verdict.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.credibility import CredibilityCheckRequest, CredibilityResponse
from app.services.credibility_service import check_credibility

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/credibility",
    tags=["Source Credibility"],
)


@router.post("/check", response_model=CredibilityResponse)
def check_source_credibility(
    payload: CredibilityCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Score the credibility of a news source by URL or domain.

    - **url**: Full URL (https://bbc.com/article) or bare domain (bbc.com).

    Returns trust score (0–100), reliability score, bias rating,
    source category, credibility label, and a one-line verdict.

    Unknown domains return `found_in_database: false` with a caution notice.
    """
    logger.info(
        "Credibility check — user=%s, url=%s",
        current_user.id,
        payload.url,
    )

    result = check_credibility(payload.url)

    logger.info(
        "Credibility result — domain=%s, label=%s, found=%s",
        result["domain"],
        result.get("credibility_label"),
        result["found_in_database"],
    )

    return CredibilityResponse(**result)
