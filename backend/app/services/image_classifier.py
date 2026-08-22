"""
Image Classification Service — thin wrapper around ResNet18 inference.

Delegates to app.ml.inference.predict_image which uses best_image_model.pth.
This service layer keeps route handlers decoupled from ML internals.

Note: image_verify_service.py calls predict_image directly for the
multimodal pipeline. This wrapper exists for standalone classification use.
"""

import logging

logger = logging.getLogger(__name__)


def classify_image(image_path: str) -> dict:
    """
    Classify an image as FAKE or REAL using the trained ResNet18 model.

    Returns:
        {
            "prediction": "FAKE" | "REAL",
            "confidence": float (0-100),
        }

    Raises:
        FileNotFoundError — model weights missing.
        Exception — image read / inference failure.
    """
    from app.ml.inference.predict_image import predict_image  # noqa: PLC0415

    result = predict_image(image_path)
    logger.info(
        "Image classification: %s (%.1f%%)",
        result["prediction"],
        result["confidence"],
    )
    return result


def classify_image_safe(image_path: str) -> dict:
    """
    Classify an image, returning None-filled fields on failure.
    Used by multimodal pipelines that must degrade gracefully.
    """
    try:
        result = classify_image(image_path)
        conf_raw = result.get("confidence", 0)
        # predict_image returns 0-100; normalise to 0-1
        confidence = conf_raw / 100.0 if conf_raw > 1.0 else conf_raw
        return {
            "image_prediction": result["prediction"],
            "image_confidence": round(confidence, 4),
            "image_class_probabilities": {
                result["prediction"]: round(confidence, 4),
            },
        }
    except Exception as exc:
        logger.warning("Image classification unavailable: %s", exc)
        return {
            "image_prediction": None,
            "image_confidence": None,
            "image_class_probabilities": {},
        }


def is_model_available() -> bool:
    """Return True if the trained ResNet18 weights exist on disk."""
    from app.ml.inference.predict_image import MODEL_PATH  # noqa: PLC0415

    return MODEL_PATH.exists()
