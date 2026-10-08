"""
Image Prediction API

POST /predict/image — standalone image classification with Grad-CAM
POST /verify/image  — full multimodal verification report with XAI
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.dependencies import get_db
from app.models.image_verification import ImageVerification
from app.models.user import User
from app.schemas.image_verify import (
    ImageClassificationResponse,
    ImageVerificationResponse,
    ImageXAIResult,
    LIMEFeature,
    SimilarArticle,
)
from app.services.image_verify_service import (
    compute_verdict,
    run_clip_branch,
    run_gradcam_branch,
    run_image_classification_branch,
    run_lime_branch,
    run_ocr_branch,
)
from app.services.prediction_service import save_prediction
from app.utils.file_utils import delete_temp_file, save_image_upload, validate_image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Verification"])

# Directory for Grad-CAM heatmap PNGs — served by StaticFiles mounted at /static
HEATMAP_DIR = Path("static/heatmaps")


def _heatmap_output_path() -> str:
    """Return a unique output path for a Grad-CAM heatmap PNG."""
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    return str(HEATMAP_DIR / filename)


# ---------------------------------------------------------------------------
# Part 2 — Standalone image classification with Grad-CAM
# ---------------------------------------------------------------------------

@router.post("/predict/image", response_model=ImageClassificationResponse)
async def classify_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image and classify it as FAKE or REAL using ResNet18.
    Also generates a Grad-CAM heatmap overlay.

    Returns HTTP 503 if the model weights are not available.
    Heatmap generation failure does not fail the prediction — heatmap_url
    will be null in that case.
    """
    t0 = time.perf_counter()
    validate_image(file)
    _, _, file_path = await save_image_upload(file)

    try:
        # --- Classification ---
        img = run_image_classification_branch(file_path)

        if img["image_prediction"] is None:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Image classifier model not trained yet. Run the training pipeline first.",
            )

        # --- Grad-CAM (non-fatal if it fails) ---
        heatmap_output = _heatmap_output_path()
        gradcam = run_gradcam_branch(file_path, heatmap_output)

        logger.info(
            "Image classification — user=%s, result=%s, gradcam=%s, total=%.3fs",
            current_user.id, img["image_prediction"],
            "ok" if gradcam["heatmap_url"] else "unavailable",
            time.perf_counter() - t0,
        )

        # confidence is 0–1 after normalisation in image_verify_service
        return ImageClassificationResponse(
            prediction=img["image_prediction"],
            confidence=img["image_confidence"],
            class_probabilities=img["image_class_probabilities"],
            heatmap_url=gradcam["heatmap_url"],
        )
    finally:
        delete_temp_file(file_path)


# ---------------------------------------------------------------------------
# Part 5 — Full multimodal verification with XAI
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

    Runs three analysis branches:
    1. OCR → text fake news prediction
    2. ResNet18 → image classification (FAKE/REAL) + Grad-CAM heatmap
    3. CLIP → FAISS reverse image search

    For lstm/bert models, LIME explanations are generated on OCR text.
    For logistic model, SHAP explanations are available via /predict endpoint.

    Returns a combined verdict with reasoning and XAI data.
    """
    t0 = time.perf_counter()
    validate_image(file)
    safe_filename, original_filename, file_path = await save_image_upload(file)

    logger.info(
        "Multimodal verification — user=%s, file=%s, model=%s",
        current_user.id, file.filename, model
    )

    try:
        # -----------------------------------------------------------------
        # Run all analysis branches
        # Each branch degrades gracefully on failure.
        # -----------------------------------------------------------------
        ocr = run_ocr_branch(file_path, model_choice=model)
        img = run_image_classification_branch(file_path)
        clip = run_clip_branch(file_path, top_k=top_k)

        # Grad-CAM — runs after image classification to reuse loaded model
        heatmap_output = _heatmap_output_path()
        gradcam = run_gradcam_branch(file_path, heatmap_output)

        # LIME — only for lstm/bert models on OCR text
        lime = run_lime_branch(ocr["ocr_text"], model_choice=model)

        logger.info(
            "All branches complete in %.3fs — ocr=%s, img=%s, clip=%d articles, "
            "gradcam=%s, lime=%s",
            time.perf_counter() - t0,
            ocr["text_prediction"],
            img["image_prediction"],
            len(clip["similar_articles"]),
            "ok" if gradcam["heatmap_url"] else "skip",
            "ok" if lime["lime_prediction"] else "skip",
        )

        # -----------------------------------------------------------------
        # Compute combined verdict
        # -----------------------------------------------------------------
        verdict, reasoning = compute_verdict(
            text_prediction=ocr["text_prediction"],
            image_prediction=img["image_prediction"],
            clip_reuse=clip["clip_reuse_detected"],
        )

        # -----------------------------------------------------------------
        # Save text prediction to DB if OCR succeeded
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # Save full verification record
        # -----------------------------------------------------------------
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
            "Verification complete — verdict=%s, user=%s, total=%.3fs",
            verdict, current_user.id, time.perf_counter() - t0,
        )

        # -----------------------------------------------------------------
        # Build XAI result
        # -----------------------------------------------------------------
        lime_features = None
        if lime["lime_explanations"] is not None:
            lime_features = [
                LIMEFeature(
                    word=e["word"],
                    impact=e["impact"],
                    direction=e["direction"],
                )
                for e in lime["lime_explanations"]
            ]

        xai_result = ImageXAIResult(
            heatmap_url=gradcam["heatmap_url"],
            gradcam_prediction=gradcam["gradcam_prediction"],
            gradcam_confidence=gradcam["gradcam_confidence"],
            lime_prediction=lime["lime_prediction"],
            lime_confidence=lime["lime_confidence"],
            lime_explanations=lime_features,
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
            xai=xai_result,
            overall_verdict=verdict,
            reasoning=reasoning,
            prediction_id=prediction_id,
            upload_id=record.id,
        )

    except Exception as exc:
        logger.error(
            "Image verification failed — user=%s, file=%s, error=%s",
            current_user.id, file.filename, exc, exc_info=True
        )
        raise
    finally:
        delete_temp_file(file_path)
