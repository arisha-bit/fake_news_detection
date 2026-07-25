"""
Reverse Image Search API — POST /images/reverse-search

Accepts an image upload, embeds it with CLIP, searches the pre-built
FAISS index, and returns visually/semantically similar known news articles.

This endpoint is separate from /upload/image (OCR + prediction).
Its sole purpose is reverse image verification — finding where an image
may have appeared before or what topics it is associated with.
"""

import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.user import User
from app.schemas.image_search import ImageMatchItem, ReverseImageResponse
from app.services.image_service import reverse_image_search
from app.utils.file_utils import (
    delete_temp_file,
    save_image_upload,
    validate_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/images",
    tags=["Reverse Image Search"],
)


@router.post("/reverse-search", response_model=ReverseImageResponse)
async def reverse_search(
    file: UploadFile = File(...),
    top_k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image and find semantically similar news articles.

    Uses CLIP to embed the image and searches against a FAISS index of
    article title embeddings in the same CLIP embedding space.

    - **file**: Image to search (JPG / JPEG / PNG, max 10 MB).
    - **top_k**: Number of results to return (default 5, max 20).

    Returns ranked matches with title, source, label, similarity score,
    and a flag indicating possible image reuse.
    """
    logger.info(
        "Reverse image search — user=%s, file=%s, top_k=%d",
        current_user.id,
        file.filename,
        top_k,
    )

    # 1. Validate
    validate_image(file)

    # 2. Save temporarily
    _, _, file_path = await save_image_upload(file)

    try:
        # 3. Run CLIP reverse search
        results, possible_reuse = reverse_image_search(file_path, top_k=top_k)

        matches = [ImageMatchItem(**item) for item in results]

        logger.info(
            "Reverse search complete — %d matches, reuse=%s",
            len(matches),
            possible_reuse,
        )

        return ReverseImageResponse(
            total_results=len(matches),
            possible_reuse_detected=possible_reuse,
            matches=matches,
        )

    finally:
        # 4. Always clean up
        delete_temp_file(file_path)
