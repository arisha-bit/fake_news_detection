"""
Upload API — POST /upload/image

Accepts multipart image uploads, extracts text via OCR,
generates Grad-CAM heatmap, and returns fake-news predictions with explainability.
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
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
from app.services.prediction_service import save_prediction
from app.services.image_verify_service import (
    run_gradcam_branch,
    run_image_classification_branch,
    run_ocr_branch,
)
from app.utils.file_utils import (
    delete_temp_file,
    save_image_upload,
    validate_image,
)

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
    model: str = Form(default="logistic"),
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
        img_result = run_image_classification_branch(file_path)
        logger.info("Image classification — %s (%.4f) in %.3fs", 
                    img_result.get("image_prediction"), 
                    img_result.get("image_confidence", 0),
                    time.perf_counter() - t0_img)

        # 5. GRAD-CAM HEATMAP (non-fatal if fails)
        t0_cam = time.perf_counter()
        heatmap_output = _heatmap_output_path()
        gradcam_result = run_gradcam_branch(file_path, heatmap_output)
        heatmap_url = gradcam_result.get("heatmap_url")
        logger.info(
            "Grad-CAM %s in %.3fs",
            "ok" if heatmap_url else "unavailable",
            time.perf_counter() - t0_cam,
        )

        # 6. OCR + text prediction (>20 meaningful chars only)
        t0_ocr = time.perf_counter()
        model_choice = model.lower()
        ocr_result = run_ocr_branch(file_path, model_choice=model_choice)
        extracted_text = ocr_result.get("ocr_text") or ""
        has_meaningful_text = ocr_result.get("text_prediction") is not None
        logger.info(
            "OCR branch — %d chars, meaningful=%s in %.3fs",
            len(extracted_text),
            has_meaningful_text,
            time.perf_counter() - t0_ocr,
        )

        # 7. TEXT EXPLAINABILITY (only when text pipeline ran)
        xai_result = None
        if has_meaningful_text:
            t0_xai = time.perf_counter()
            try:
                xai_result = generate_model_explanation(
                    text=extracted_text,
                    model_choice=model_choice,
                    method="shap" if model_choice == "logistic" else "lime",
                    top_k=10,
                )
                logger.info(
                    "Explainability generated (%s) in %.3fs",
                    xai_result.get("explanation_type", "unknown"),
                    time.perf_counter() - t0_xai,
                )
            except Exception as e:
                logger.warning("Explainability generation failed (non-fatal): %s", e)

        # 8. KEYWORDS & CLICKBAIT (traditional NLP)
        keywords = extract_keywords(extracted_text) if extracted_text.strip() else []
        clickbait = clickbait_score(extracted_text) if extracted_text.strip() else 0
        if has_meaningful_text:
            explanation = generate_explanation(
                ocr_result["text_prediction"],
                clickbait,
            )
        else:
            explanation = (
                "Image classified independently. "
                "OCR did not yield enough text for a separate text-model prediction."
            )

        # 9. Save text prediction record to DB (meaningful OCR only)
        saved_prediction = None
        if has_meaningful_text:
            saved_prediction = save_prediction(
                db=db,
                user_id=current_user.id,
                text=extracted_text,
                prediction=ocr_result["text_prediction"],
                confidence=ocr_result["text_confidence"] or 0.0,
                model_name=model_choice,
            )
            uploaded_file_record.prediction_id = saved_prediction.id
            db.commit()
            db.refresh(uploaded_file_record)

        logger.info(
            "Upload complete — img=%s, text=%s, heatmap=%s, xai=%s",
            img_result.get("image_prediction"),
            ocr_result.get("text_prediction"),
            "ok" if heatmap_url else "unavailable",
            "ok" if xai_result else "unavailable",
        )

        return UploadResponse(
            image_prediction=img_result.get("image_prediction", "UNAVAILABLE"),
            image_confidence=img_result.get("image_confidence", 0.0),
            heatmap_url=heatmap_url,
            extracted_text=extracted_text,
            has_meaningful_text=has_meaningful_text,
            text_prediction=ocr_result.get("text_prediction"),
            text_confidence=ocr_result.get("text_confidence"),
            keywords=keywords,
            clickbait_score=clickbait,
            explanation=explanation,
            xai=xai_result,
            prediction_id=saved_prediction.id if saved_prediction else None,
            uploaded_file_id=uploaded_file_record.id,
        )

    finally:
        # 12. Always clean up the temp file from disk
        delete_temp_file(file_path)
