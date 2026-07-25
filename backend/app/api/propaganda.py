"""
Propaganda Detection API — POST /propaganda/analyse

Accepts article text and returns a structured analysis of propaganda
and manipulation techniques present in the content.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.propaganda import (
    PropagandaRequest,
    PropagandaResponse,
    PropagandaTechnique,
)
from app.services.propaganda_service import detect_propaganda

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/propaganda",
    tags=["Propaganda Detection"],
)


@router.post("/analyse", response_model=PropagandaResponse)
def analyse_propaganda(
    payload: PropagandaRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyse text for propaganda and manipulation techniques.

    Detects: Fear Appeal, Clickbait, Loaded Language, Conspiracy Framing,
    Emotional Manipulation, Bandwagon, False Dilemma, Name Calling,
    Glittering Generalities, Repetition.

    Returns per-technique confidence scores, matched phrases,
    and an overall propaganda score.
    """
    logger.info(
        "Propaganda analysis — user=%s, text_len=%d",
        current_user.id,
        len(payload.text),
    )

    result = detect_propaganda(payload.text)

    return PropagandaResponse(
        propaganda_detected=result["propaganda_detected"],
        overall_score=result["overall_score"],
        techniques_found=[
            PropagandaTechnique(**t) for t in result["techniques_found"]
        ],
        summary=result["summary"],
    )
