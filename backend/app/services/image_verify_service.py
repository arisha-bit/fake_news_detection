"""
Image Verification Orchestration Service — Part 5.

Combines three independent analysis branches:
  1. OCR branch  — EasyOCR + existing text predictor (DistilBERT/logistic/LSTM)
  2. Image branch — EfficientNetB0 image classifier
  3. CLIP branch  — FAISS reverse image search

Each branch runs independently. If one fails it degrades gracefully
(returns None for that branch) rather than failing the whole request.

Final verdict logic:
  - Both FAKE → LIKELY FAKE
  - Both REAL → LIKELY REAL
  - Image FAKE + Text REAL → LIKELY MISLEADING (image manipulation)
  - Image REAL + Text FAKE → LIKELY FAKE (misleading caption/text)
  - Any branch unavailable → UNCERTAIN
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def run_ocr_branch(image_path: str, model_choice: str = "logistic") -> dict:
    """
    Run OCR on the image then predict using the text pipeline.
    Returns dict with ocr_text, prediction, confidence.
    Returns None-filled dict on failure.
    """
    try:
        from app.services.ocr_service import extract_text_from_image  # noqa
        from app.api.prediction import _run_model  # noqa

        ocr_text = extract_text_from_image(image_path)
        result = _run_model(model_choice, ocr_text)

        logger.info("OCR branch: %s (%.2f%%)", result["prediction"], result["confidence"] * 100)
        return {
            "ocr_text": ocr_text,
            "text_prediction": result["prediction"],
            "text_confidence": result["confidence"],
        }
    except Exception as exc:
        logger.warning("OCR branch failed: %s", exc)
        return {
            "ocr_text": "",
            "text_prediction": None,
            "text_confidence": None,
        }


def run_image_classification_branch(image_path: str) -> dict:
    """
    Run EfficientNetB0 image classifier on the image.
    Returns dict with image_prediction, image_confidence, class_probabilities.
    Returns None-filled dict if model not trained yet.
    """
    try:
        from app.ml.image_classifier.predict import predict_image  # noqa

        result = predict_image(image_path)
        logger.info(
            "Image branch: %s (%.2f%%)",
            result["prediction"], result["confidence"] * 100
        )
        return {
            "image_prediction": result["prediction"],
            "image_confidence": result["confidence"],
            "image_class_probabilities": result["class_probabilities"],
        }
    except Exception as exc:
        logger.warning("Image classification branch failed (model may not be trained): %s", exc)
        return {
            "image_prediction": None,
            "image_confidence": None,
            "image_class_probabilities": {},
        }


def run_clip_branch(image_path: str, top_k: int = 5) -> dict:
    """
    Run CLIP reverse image search against the FAISS index.
    Returns dict with similar_articles and reuse_detected.
    """
    try:
        from app.services.image_service import reverse_image_search  # noqa

        results, reuse = reverse_image_search(image_path, top_k=top_k)
        logger.info("CLIP branch: %d matches, reuse=%s", len(results), reuse)
        return {
            "similar_articles": results,
            "clip_reuse_detected": reuse,
        }
    except Exception as exc:
        logger.warning("CLIP branch failed: %s", exc)
        return {
            "similar_articles": [],
            "clip_reuse_detected": False,
        }


def compute_verdict(
    text_prediction: Optional[str],
    image_prediction: Optional[str],
    clip_reuse: bool,
) -> tuple[str, list[str]]:
    """
    Combine branch results into an overall verdict and reasoning list.

    Returns:
        (verdict_string, [reasoning_sentence, ...])
    """
    reasoning = []
    both_available = text_prediction is not None and image_prediction is not None

    if not both_available:
        if text_prediction == "FAKE":
            reasoning.append("OCR text analysis detected fake news patterns.")
            return "LIKELY FAKE", reasoning
        if image_prediction == "FAKE":
            reasoning.append("Image classification detected image manipulation.")
            return "LIKELY FAKE", reasoning
        reasoning.append("Insufficient data for a conclusive verdict.")
        return "UNCERTAIN", reasoning

    # Both available
    if text_prediction == "FAKE" and image_prediction == "FAKE":
        reasoning.append("Both OCR text analysis and image classification detected fake signals.")
        if clip_reuse:
            reasoning.append("Image appears in known misleading news contexts.")
        return "LIKELY FAKE", reasoning

    if text_prediction == "REAL" and image_prediction == "REAL":
        reasoning.append("Both OCR text and image analysis indicate authentic content.")
        if clip_reuse:
            reasoning.append("Similar images found in known news — consistent with real reporting.")
        return "LIKELY REAL", reasoning

    if image_prediction == "FAKE" and text_prediction == "REAL":
        reasoning.append("Image appears manipulated or out-of-context despite neutral text.")
        if clip_reuse:
            reasoning.append("Image has been used in other news contexts — possible misuse.")
        return "LIKELY MISLEADING", reasoning

    if image_prediction == "REAL" and text_prediction == "FAKE":
        reasoning.append("Image appears authentic but accompanying text contains fake news patterns.")
        return "LIKELY FAKE", reasoning

    reasoning.append("Analysis did not produce a conclusive signal.")
    return "UNCERTAIN", reasoning
