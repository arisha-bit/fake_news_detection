"""
Upload API — POST /upload/image

Accepts multipart image uploads, extracts text via OCR,
generates Grad-CAM heatmap, and returns fake-news predictions with explainability.
"""

import logging
import time
import uuid
from pathlib import Path

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
    generate_model_explanation,
)
from app.services.ocr_service import extract_text_from_image
from app.services.prediction_service import save_prediction
from app.services.image_verify_service import run_gradcam_branch
from app.utils.file_utils import (
    delete_temp_file,
    save_image_upload,
    validate_image,
)

# Reuse existing model runner — no prediction logic duplicated
from app.api.prediction import _run_model

logger = logging.getLogger(__name__)

# Directory for Grad-CAM heatmap PNGs — served by StaticFiles mounted at /static
HEATMAP_DIR = Path("static/heatmaps")


def _heatmap_output_path() -> str:
    """Return a unique output path for a Grad-CAM heatmap PNG."""
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    return str(HEATMAP_DIR / filename)

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
    Upload an image, classify it with ResNet18, extract text via OCR,
    generate Grad-CAM heatmap, and predict fake-news using text model.
    
    Returns predictions from both image and text classifiers, plus explainability.

    - **file**: multipart image upload (JPG/JPEG/PNG)
    - **model**: text inference model — `logistic` (default), `lstm`, or `bert`
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

        # 4. IMAGE CLASSIFICATION via ResNet18
        t0_img = time.perf_counter()
        from app.services.image_verify_service import run_image_classification_branch
        img_result = run_image_classification_branch(file_path)
        logger.info("Image classification — %s (%.4f) in %.3fs", 
                    img_result.get("image_prediction"), 
                    img_result.get("image_confidence", 0),
                    time.perf_counter() - t0_img)

        # 5. GRAD-CAM HEATMAP (non-fatal if fails)
        t0_cam = time.perf_counter()
        heatmap_url = None
        try:
            heatmap_output = _heatmap_output_path()
            gradcam_result = run_gradcam_branch(file_path, heatmap_output)
            if gradcam_result and gradcam_result.get("heatmap_path"):
                # Convert disk path to URL
                heatmap_url = f"/static/heatmaps/{Path(gradcam_result['heatmap_path']).name}"
                logger.info("Grad-CAM generated in %.3fs", time.perf_counter() - t0_cam)
        except Exception as e:
            logger.warning("Grad-CAM generation failed (non-fatal): %s", e)

        # 6. OCR — extract text from image
        t0_ocr = time.perf_counter()
        extracted_text = extract_text_from_image(file_path)
        logger.info("OCR completed — %d chars extracted in %.3fs", 
                    len(extracted_text), time.perf_counter() - t0_ocr)

        # 7. TEXT PREDICTION via selected model (logistic/lstm/bert)
        t0_text = time.perf_counter()
        model_choice = model.lower()
        text_result = _run_model(model_choice, extracted_text)
        logger.info("Text prediction — %s (%.4f) in %.3fs", 
                    text_result["prediction"], text_result["confidence"],
                    time.perf_counter() - t0_text)

        # 8. TEXT EXPLAINABILITY (SHAP/LIME based on model)
        t0_xai = time.perf_counter()
        xai_result = None
        try:
            xai_result = generate_model_explanation(
                text=extracted_text,
                model_choice=model_choice,
                method="shap" if model_choice == "logistic" else "lime",
                top_k=10,
            )
            logger.info("Explainability generated (%s) in %.3fs", 
                        xai_result.get("explanation_type", "unknown"),
                        time.perf_counter() - t0_xai)
        except Exception as e:
            logger.warning("Explainability generation failed (non-fatal): %s", e)

        # 9. KEYWORDS & CLICKBAIT (traditional NLP)
        keywords = extract_keywords(extracted_text)
        clickbait = clickbait_score(extracted_text)
        explanation = generate_explanation(text_result["prediction"], clickbait)

        # 10. Save text prediction record to DB
        saved_prediction = save_prediction(
            db=db,
            user_id=current_user.id,
            text=extracted_text,
            prediction=text_result["prediction"],
            confidence=text_result["confidence"],
            model_name=model_choice,
        )

        # 11. Link prediction back to uploaded file record
        uploaded_file_record.prediction_id = saved_prediction.id
        db.commit()
        db.refresh(uploaded_file_record)

        logger.info(
            "Upload complete — img=%s, text=%s, heatmap=%s, xai=%s",
            img_result.get("image_prediction"),
            text_result["prediction"],
            "ok" if heatmap_url else "unavailable",
            "ok" if xai_result else "unavailable",
        )

        return UploadResponse(
            image_prediction=img_result.get("image_prediction", "UNAVAILABLE"),
            image_confidence=img_result.get("image_confidence", 0.0),
            heatmap_url=heatmap_url,
            extracted_text=extracted_text,
            text_prediction=text_result["prediction"],
            text_confidence=text_result["confidence"],
            keywords=keywords,
            clickbait_score=clickbait,
            explanation=explanation,
            xai=xai_result,
            prediction_id=saved_prediction.id,
            uploaded_file_id=uploaded_file_record.id,
        )

    finally:
        # 12. Always clean up the temp file from disk
        delete_temp_file(file_path)
