"""
Image Verification Orchestration Service.

Combines three independent analysis branches:
  1. OCR branch  — EasyOCR + existing text predictor (logistic/LSTM/DistilBERT)
  2. Image branch — ResNet18 image classifier (best_image_model.pth)
  3. CLIP branch  — FAISS reverse image search

Each branch runs independently. If one fails it degrades gracefully
(returns None for that branch) rather than failing the whole request.

ResNet18 is the ONLY image classifier used here.
Do NOT use EfficientNetB0 or image_classifier.pt.

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

    OCR failure does NOT fail the whole request — returns empty/None on error.
    Only meaningful text (>20 chars) is passed to prediction.

    Returns:
        {ocr_text, text_prediction, text_confidence}
    """
    try:
        from app.services.ocr_service import extract_text_from_image
        from app.api.prediction import _run_model

        ocr_text = extract_text_from_image(image_path)

        # Only predict if meaningful text was found
        if not ocr_text or len(ocr_text.strip()) < 20:
            logger.info("OCR branch: insufficient text extracted (%d chars)", len(ocr_text or ""))
            return {
                "ocr_text": ocr_text or "",
                "text_prediction": None,
                "text_confidence": None,
            }

        result = _run_model(model_choice, ocr_text)
        logger.info(
            "OCR branch: %s (%.2f%%)",
            result["prediction"],
            result["confidence"] * 100,
        )
        return {
            "ocr_text": ocr_text,
            "text_prediction": result["prediction"],
            "text_confidence": result["confidence"],
        }
    except Exception as exc:
        logger.warning("OCR branch failed (non-fatal): %s", exc)
        return {
            "ocr_text": "",
            "text_prediction": None,
            "text_confidence": None,
        }


def run_image_classification_branch(image_path: str) -> dict:
    """
    Run ResNet18 image classifier on the image.

    Uses best_image_model.pth via the existing predict_image inference module.
    Returns None-filled dict if model not available.

    Returns:
        {image_prediction, image_confidence, image_class_probabilities}
    """
    try:
        from app.ml.inference.predict_image import predict_image

        result = predict_image(image_path)

        # predict_image returns confidence as 0-100 float — normalise to 0-1
        raw_conf = result.get("confidence", 0)
        confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf

        prediction = result.get("prediction", "UNKNOWN")

        logger.info(
            "ResNet18 branch: %s (%.2f%%)", prediction, confidence * 100
        )

        return {
            "image_prediction": prediction,
            "image_confidence": round(confidence, 4),
            "image_class_probabilities": {
                prediction: round(confidence, 4),
            },
        }
    except Exception as exc:
        logger.warning(
            "Image classification branch unavailable (model may not exist): %s", exc
        )
        return {
            "image_prediction": None,
            "image_confidence": None,
            "image_class_probabilities": {},
        }


def run_clip_branch(image_path: str, top_k: int = 5) -> dict:
    """
    Run CLIP reverse image search against the FAISS index.

    CLIP similarity results represent semantic/visual similarity to indexed
    news articles. They are retrieval evidence — NOT proof of image reuse
    or misinformation on their own.

    Returns:
        {similar_articles, clip_reuse_detected}
    """
    try:
        from app.services.image_service import reverse_image_search

        results, reuse = reverse_image_search(image_path, top_k=top_k)
        logger.info(
            "CLIP branch: %d similar articles found, high_similarity=%s",
            len(results),
            reuse,
        )
        return {
            "similar_articles": results,
            "clip_reuse_detected": reuse,
        }
    except Exception as exc:
        logger.warning("CLIP branch failed (non-fatal): %s", exc)
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
    Combine branch results into a cautious overall verdict and reasoning list.

    Reasoning language is calibrated to reflect actual model evidence
    without making claims stronger than the evidence supports.

    Returns:
        (verdict_string, [reasoning_sentence, ...])
    """
    reasoning: list[str] = []
    both_available = text_prediction is not None and image_prediction is not None

    if not both_available:
        if text_prediction == "FAKE":
            reasoning.append(
                "OCR text analysis detected linguistic patterns associated with fake news."
            )
            return "LIKELY FAKE", reasoning

        if image_prediction == "FAKE":
            reasoning.append(
                "Image classifier detected visual patterns associated with misleading content."
            )
            return "LIKELY FAKE", reasoning

        reasoning.append(
            "Only one analysis branch returned results. "
            "Verdict is uncertain without corroborating evidence."
        )
        return "UNCERTAIN", reasoning

    # Both branches available
    if text_prediction == "FAKE" and image_prediction == "FAKE":
        reasoning.append(
            "Both OCR text analysis and image classification indicate fake news signals."
        )
        if clip_reuse:
            reasoning.append(
                "Visually similar images appear in indexed news content — "
                "retrieved for reference only."
            )
        return "LIKELY FAKE", reasoning

    if text_prediction == "REAL" and image_prediction == "REAL":
        reasoning.append(
            "Both OCR text and image analysis indicate content consistent with real news."
        )
        if clip_reuse:
            reasoning.append(
                "Similar images found in indexed news — "
                "consistent with genuine news reporting."
            )
        return "LIKELY REAL", reasoning

    if image_prediction == "FAKE" and text_prediction == "REAL":
        reasoning.append(
            "Image classifier flagged the image, but OCR text analysis found neutral language. "
            "The image may be out-of-context or the text content may be accurate."
        )
        if clip_reuse:
            reasoning.append(
                "Similar images retrieved from indexed news — "
                "context should be verified independently."
            )
        return "LIKELY MISLEADING", reasoning

    if image_prediction == "REAL" and text_prediction == "FAKE":
        reasoning.append(
            "Image appears visually authentic, but text analysis detected "
            "linguistic patterns associated with fake news."
        )
        return "LIKELY FAKE", reasoning

    reasoning.append(
        "Analysis branches produced inconclusive or conflicting signals."
    )
    return "UNCERTAIN", reasoning
