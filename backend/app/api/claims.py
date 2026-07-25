"""
Claims API — POST /claims/extract

Accepts article text, splits it into factual claims,
runs each claim through the existing prediction pipeline,
and returns per-claim verdicts plus an overall majority-vote result.

No prediction logic is duplicated — _run_model() is reused directly.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.prediction import _run_model
from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.claim import (
    ClaimExtractionRequest,
    ClaimExtractionResponse,
    ClaimResult,
)
from app.services.claim_service import compute_overall_verdict, extract_claims

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/claims",
    tags=["Claims"],
)


@router.post("/extract", response_model=ClaimExtractionResponse)
def extract_and_verify_claims(
    payload: ClaimExtractionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Split article text into factual claims and verify each independently.

    - **text**: Full article text.
    - **model**: `logistic` (default), `lstm`, or `bert`.

    Returns per-claim predictions and an overall majority-vote verdict.
    """
    logger.info(
        "Claim extraction started — user=%s, model=%s, text_len=%d",
        current_user.id,
        payload.model,
        len(payload.text),
    )

    # 1. Extract claims
    claims = extract_claims(payload.text)

    if not claims:
        raise HTTPException(
            status_code=422,
            detail="No verifiable claims could be extracted from the provided text. "
                   "Ensure the article contains complete factual sentences.",
        )

    model_choice = payload.model.lower()

    # 2. Verify each claim independently
    results: list[ClaimResult] = []
    predictions: list[str] = []
    confidences: list[float] = []

    for idx, claim_text in enumerate(claims):
        result = _run_model(model_choice, claim_text)

        results.append(
            ClaimResult(
                claim_index=idx,
                text=claim_text,
                prediction=result["prediction"],
                confidence=result["confidence"],
            )
        )
        predictions.append(result["prediction"])
        confidences.append(result["confidence"])

    # 3. Compute overall verdict via majority vote
    overall_verdict, overall_confidence = compute_overall_verdict(
        predictions, confidences
    )

    fake_count = predictions.count("FAKE")
    real_count = predictions.count("REAL")

    logger.info(
        "Claim verification complete — %d claims, verdict=%s, confidence=%.4f",
        len(results),
        overall_verdict,
        overall_confidence,
    )

    return ClaimExtractionResponse(
        total_claims=len(results),
        fake_claims=fake_count,
        real_claims=real_count,
        overall_verdict=overall_verdict,
        overall_confidence=overall_confidence,
        claims=results,
    )
