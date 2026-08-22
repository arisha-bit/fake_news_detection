"""
Upload API — POST /upload/image

Accepts multipart image uploads, extracts text via OCR,
and returns fake-news predictions using the existing pipeline.
"""

import logging

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.uploaded_file import UploadedFile
from app.models.user import User
from app.schemas.upload import UploadResponse
from app.services.explainability_service import (
    clickbait_score,
    extract_keywords,
    generate_explanation,
)
from app.services.ocr_service import extract_text_from_image
from app.services.prediction_service import save_prediction
from app.utils.file_utils import (
    delete_temp_file,
    save_image_upload,
    validate_image,
)

# Reuse existing model runner — no prediction logic duplicated
from app.api.prediction import _run_model

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


@router.post("/image", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    model: str = "logistic",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image (JPG / JPEG / PNG), extract text via OCR,
    and return a fake-news prediction.

    - **file**: multipart image upload
    - **model**: inference model — `logistic` (default), `lstm`, or `bert`
    """
    logger.info(
        "Upload started — user=%s, file=%s, model=%s",
        current_user.id,
        file.filename,
        model,
    )

    # 1. Validate extension + MIME type before touching disk
    validate_image(file)

    # 2. Persist file to uploads/images/
    safe_filename, original_filename, file_path = await save_image_upload(file)

    uploaded_file_record: UploadedFile | None = None

    try:
        # 3. Persist file metadata to DB immediately so it's traceable
        uploaded_file_record = UploadedFile(
            user_id=current_user.id,
            filename=safe_filename,
            original_filename=original_filename,
            file_type=file.content_type or "image/jpeg",
            file_path=file_path,
        )
        db.add(uploaded_file_record)
        db.commit()
        db.refresh(uploaded_file_record)

        # 4. OCR — extract text from image
        extracted_text = extract_text_from_image(file_path)
        logger.info("OCR completed — %d chars extracted", len(extracted_text))

        # 5. Run prediction via existing service (no logic duplicated)
        model_choice = model.lower()
        result = _run_model(model_choice, extracted_text)

        # 6. Explainability
        keywords = extract_keywords(extracted_text)
        score = clickbait_score(extracted_text)
        explanation = generate_explanation(result["prediction"], score)

        # 7. Save prediction record
        saved_prediction = save_prediction(
            db=db,
            user_id=current_user.id,
            text=extracted_text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_name=model_choice,
        )

        # 8. Link prediction back to uploaded file record
        uploaded_file_record.prediction_id = saved_prediction.id
        db.commit()
        db.refresh(uploaded_file_record)

        logger.info(
            "Prediction completed — result=%s, confidence=%.4f",
            result["prediction"],
            result["confidence"],
        )

        return UploadResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            extracted_text=extracted_text,
            keywords=keywords,
            clickbait_score=score,
            explanation=explanation,
            prediction_id=saved_prediction.id,
            uploaded_file_id=uploaded_file_record.id,
        )

    finally:
        # 9. Always clean up the temp file from disk
        delete_temp_file(file_path)
