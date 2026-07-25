"""
Evidence Retrieval API — POST /evidence/search

Accepts a query (claim or article) and returns top-k semantically similar
articles from the trusted corpus with similarity scores and labels.

Designed to work standalone or as a companion to /claims/extract:
  - Pass the full article for article-level evidence.
  - Pass individual claim texts for claim-level evidence.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.evidence import (
    EvidenceItem,
    EvidenceSearchRequest,
    EvidenceSearchResponse,
)
from app.services.retrieval_service import search_evidence

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/evidence",
    tags=["Evidence"],
)


@router.post("/search", response_model=EvidenceSearchResponse)
def search(
    payload: EvidenceSearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Search for semantically similar articles from the trusted corpus.

    - **query**: Claim or article text to search evidence for.
    - **top_k**: Number of results to return (default 5, max 20).

    Returns ranked evidence items with title, snippet, similarity score,
    and the ground-truth label (REAL/FAKE) from the training corpus.
    """
    logger.info(
        "Evidence search — user=%s, top_k=%d, query_len=%d",
        current_user.id,
        payload.top_k,
        len(payload.query),
    )

    raw_results = search_evidence(payload.query, top_k=payload.top_k)

    evidence_items = [EvidenceItem(**item) for item in raw_results]

    return EvidenceSearchResponse(
        query=payload.query,
        total_results=len(evidence_items),
        evidence=evidence_items,
    )
