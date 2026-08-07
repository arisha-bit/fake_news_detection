"""
Image Prediction API

Part 2: POST /predict/image — standalone image classification
Part 5: POST /verify/image  — full multimodal verification report

Part 3 (OCR): POST /upload/image already exists — NOT modified.
Part 4 (CLIP): POST /images/reverse-search already exists — NOT modified.
"""

import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.image_verification import ImageVerification
from app.models.user import User
from app.schemas.image_verify import ImageClassificationResponse, ImageVerificationResponse, SimilarArticle
from app.services.image_verify_service import (
    compute_verdict,
    run_clip_branch,
    run_image_classification_branch,
    run_ocr_branch,
)
from app.services.prediction_service import save_prediction
from app.utils.file_utils import delete_temp_file, save_image_upload, validate_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Verification"])


# ---------------------------------------------------------------------------
# Part 2 — Standalone image classification
# ---------------------------------------------------------------------------

@router.post("/predict/image", response_model=ImageClassificationResponse)
async def classify_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image and classify it as FAKE or REAL using the trained
    EfficientNetB0 image classifier (visual features only, no OCR).

    Requires training the model first:
        python -m app.ml.image_classifier.train --data_dir <dataset_path>
    """
    validate_image(file)
    _, _, file_path = await save_image_upload(file)

    try:
        result = run_image_classification_branch(file_path)

        if result["image_prediction"] is None:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Image classifier model not trained yet. Run the training pipeline first.",
            )

        logger.info(
            "Image classification — user=%s, result=%s",
            current_user.id, result["image_prediction"]
        )

        return ImageClassificationResponse(
            prediction=result["image_prediction"],
            confidence=result["image_confidence"],
            class_probabilities=result["image_class_probabilities"],
        )
    finally:
        delete_temp_file(file_path)


# ---------------------------------------------------------------------------
# Part 5 — Full multimodal verification
# ---------------------------------------------------------------------------

@router.post("/verify/image", response_model=ImageVerificationResponse)
async def verify_image(
    file: UploadFile = File(...),
    model: str = Query(default="logistic", description="Text model: logistic | lstm | bert"),
    top_k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Full multimodal image verification.

    Runs three parallel analysis branches:
    1. OCR → text fake news prediction
    2. EfficientNetB0 → image classification (FAKE/REAL)
    3. CLIP → FAISS reverse image search

    Returns a combined verdict with reasoning.
    """
    validate_image(file)
    safe_filename, original_filename, file_path = await save_image_upload(file)

    logger.info(
        "Multimodal verification — user=%s, file=%s, model=%s",
        current_user.id, file.filename, model
    )

    try:
        # Run all three branches (each degrades gracefully on failure)
        ocr = run_ocr_branch(file_path, model_choice=model)
        img = run_image_classification_branch(file_path)
        clip = run_clip_branch(file_path, top_k=top_k)

        # Compute combined verdict
        verdict, reasoning = compute_verdict(
            text_prediction=ocr["text_prediction"],
            image_prediction=img["image_prediction"],
            clip_reuse=clip["clip_reuse_detected"],
        )

        # Save text prediction to DB if OCR succeeded
        prediction_id = None
        if ocr["text_prediction"] and ocr["ocr_text"]:
            saved = save_prediction(
                db=db,
                user_id=current_user.id,
                text=ocr["ocr_text"],
                prediction=ocr["text_prediction"],
                confidence=ocr["text_confidence"] or 0.0,
                model_name=model,
            )
            prediction_id = saved.id

        # Save full verification record
        record = ImageVerification(
            user_id=current_user.id,
            filename=safe_filename,
            original_filename=original_filename,
            file_path=file_path,
            ocr_text=ocr["ocr_text"],
            text_prediction=ocr["text_prediction"],
            text_confidence=ocr["text_confidence"],
            image_prediction=img["image_prediction"],
            image_confidence=img["image_confidence"],
            clip_reuse_detected=clip["clip_reuse_detected"],
            overall_verdict=verdict,
            model_used=model,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "Verification complete — verdict=%s, user=%s",
            verdict, current_user.id
        )

        return ImageVerificationResponse(
            ocr_text=ocr["ocr_text"],
            text_prediction=ocr["text_prediction"] or "UNAVAILABLE",
            text_confidence=ocr["text_confidence"] or 0.0,
            image_prediction=img["image_prediction"] or "UNAVAILABLE",
            image_confidence=img["image_confidence"] or 0.0,
            image_class_probabilities=img["image_class_probabilities"],
            similar_articles=[SimilarArticle(**a) for a in clip["similar_articles"]],
            clip_reuse_detected=clip["clip_reuse_detected"],
            overall_verdict=verdict,
            reasoning=reasoning,
            prediction_id=prediction_id,
            upload_id=record.id,
        )

    finally:
        delete_temp_file(file_path)
