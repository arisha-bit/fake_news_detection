"""
Image Verification Orchestration Service.

Combines three independent analysis branches:
  1. OCR branch  — EasyOCR + existing text predictor (logistic/LSTM/DistilBERT)
  2. Image branch — ResNet18 image classifier (best_image_model.pth)
  3. CLIP branch  — FAISS reverse image search

Each branch runs independently. If one fails it degrades gracefully
(returns None for that branch) rather than failing the whole request.

ResNet18 is the ONLY image classifier used here.

Final verdict logic:
  - Both FAKE → LIKELY FAKE
  - Both REAL → LIKELY REAL
  - Image FAKE + Text REAL → LIKELY MISLEADING (image manipulation)
  - Image REAL + Text FAKE → LIKELY FAKE (misleading caption/text)
  - Any branch unavailable → UNCERTAIN
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def run_ocr_branch(image_path: str, model_choice: str = "logistic") -> dict:
    """
    Run OCR on the image then predict using the text pipeline.

    OCR returns empty string when no text found — this is NOT an error.
    Only runs prediction if meaningful text (>20 chars) is found.

    Returns:
        {ocr_text, text_prediction, text_confidence}
    """
    t0 = time.perf_counter()
    try:
        from app.services.ocr_service import extract_text_from_image
        from app.api.prediction import _run_model

        ocr_text = extract_text_from_image(image_path)
        logger.info("OCR branch: extracted %d chars in %.3fs", len(ocr_text or ""), time.perf_counter() - t0)

        if not ocr_text or len(ocr_text.strip()) < 20:
            logger.info(
                "OCR branch: no meaningful text extracted (%d chars) — skipping text analysis",
                len(ocr_text or ""),
            )
            return {
                "ocr_text": ocr_text or "",
                "text_prediction": None,
                "text_confidence": None,
            }

        result = _run_model(model_choice, ocr_text)
        logger.info(
            "OCR branch: %s (%.2f%%) total=%.3fs",
            result["prediction"],
            result["confidence"] * 100,
            time.perf_counter() - t0,
        )
        return {
            "ocr_text": ocr_text,
            "text_prediction": result["prediction"],
            "text_confidence": result["confidence"],
        }
    except Exception as exc:
        logger.warning("OCR branch failed (non-fatal): %s", exc, exc_info=True)
        return {
            "ocr_text": "",
            "text_prediction": None,
            "text_confidence": None,
        }


def run_image_classification_branch(image_path: str) -> dict:
    """
    Run ResNet18 image classifier on the image.

    Uses best_image_model.pth via the shared predict_image inference module.
    Returns None-filled dict if model not available.

    Returns:
        {image_prediction, image_confidence, image_class_probabilities}
        where image_class_probabilities contains BOTH "FAKE" and "REAL" probabilities.
    """
    t0 = time.perf_counter()
    try:
        from app.ml.inference.predict_image import predict_image

        result = predict_image(image_path)

        # predict_image returns confidence as 0-100 float — normalise to 0-1
        raw_conf = result.get("confidence", 0)
        confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf

        prediction = result.get("prediction", "UNKNOWN")

        # Use both-class probabilities from the updated predict_image
        probabilities = result.get("probabilities", {})
        if not probabilities:
            # Fallback: construct from confidence alone (old API compat)
            other_conf = round(1.0 - confidence, 4)
            other_class = "REAL" if prediction == "FAKE" else "FAKE"
            probabilities = {
                prediction: round(confidence, 4),
                other_class: other_conf,
            }

        logger.info(
            "ResNet18 branch: %s (%.2f%%) in %.3fs", prediction, confidence * 100, time.perf_counter() - t0
        )

        return {
            "image_prediction": prediction,
            "image_confidence": round(confidence, 4),
            "image_class_probabilities": probabilities,
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

    Returns:
        {similar_articles, clip_reuse_detected}
    """
    t0 = time.perf_counter()
    try:
        from app.services.image_service import reverse_image_search

        results, reuse = reverse_image_search(image_path, top_k=top_k)
        logger.info(
            "CLIP branch: %d similar articles, high_similarity=%s in %.3fs",
            len(results), reuse, time.perf_counter() - t0,
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


def run_gradcam_branch(image_path: str, output_path: str) -> dict:
    """
    Generate Grad-CAM heatmap for the image using the shared ResNet18 model.

    Runs AFTER image classification to reuse already-loaded model.
    Failures are non-fatal — returns None fields so the rest of the response
    is unaffected.

    Args:
        image_path:  Path to the source image.
        output_path: Disk path where the PNG will be saved (e.g. "static/heatmaps/abc.png").
                     Must be under the "static/" directory so it can be served.

    Returns:
        {
            "gradcam_prediction": str | None,
            "gradcam_confidence": float | None,
            "heatmap_path": str | None,       # disk path
            "heatmap_url": str | None,        # URL path like /static/heatmaps/abc.png
        }
    """
    t0 = time.perf_counter()
    try:
        from app.ml.explainability.gradcam_image import generate_gradcam

        result = generate_gradcam(image_path, output_path)

        # Build the URL: output_path is "static/heatmaps/<uuid>.png"
        # URL should be "/static/heatmaps/<uuid>.png"
        heatmap_url = "/" + output_path.replace("\\", "/").lstrip("/")

        logger.info(
            "Grad-CAM branch: %s (%.2f) in %.3fs, url=%s",
            result["prediction"], result["confidence"],
            time.perf_counter() - t0, heatmap_url,
        )

        return {
            "gradcam_prediction": result["prediction"],
            "gradcam_confidence": result["confidence"],
            "heatmap_path": result["heatmap_path"],
            "heatmap_url": heatmap_url,
        }

    except FileNotFoundError as exc:
        logger.warning("Grad-CAM skipped — model weights missing: %s", exc)
        return {"gradcam_prediction": None, "gradcam_confidence": None, "heatmap_path": None, "heatmap_url": None}
    except ImportError as exc:
        logger.warning("Grad-CAM skipped — pytorch_grad_cam not installed: %s", exc)
        return {"gradcam_prediction": None, "gradcam_confidence": None, "heatmap_path": None, "heatmap_url": None}
    except Exception as exc:
        logger.warning("Grad-CAM branch failed (non-fatal): %s", exc, exc_info=True)
        return {"gradcam_prediction": None, "gradcam_confidence": None, "heatmap_path": None, "heatmap_url": None}


def run_lime_branch(text: str, model_choice: str) -> dict:
    """
    Run LIME explanation for lstm or bert models on OCR-extracted text.

    SHAP is used for logistic regression (handled in prediction.py).
    LIME is used here for lstm and bert when OCR text is available.

    Returns:
        {
            "lime_prediction": str | None,
            "lime_confidence": float | None,
            "lime_explanations": list | None,
        }
    """
    if not text or len(text.strip()) < 20:
        return {"lime_prediction": None, "lime_confidence": None, "lime_explanations": None}

    t0 = time.perf_counter()
    try:
        if model_choice == "lstm":
            from app.ml.explainability.lime_lstm import explain_lstm
            result = explain_lstm(text, num_features=10)
        elif model_choice == "bert":
            from app.ml.explainability.lime_bert import explain_bert
            result = explain_bert(text, num_features=10)
        else:
            # logistic uses SHAP, not LIME
            return {"lime_prediction": None, "lime_confidence": None, "lime_explanations": None}

        logger.info(
            "LIME branch (%s): %s (%.2f) in %.3fs",
            model_choice, result["prediction"], result["confidence"],
            time.perf_counter() - t0,
        )
        return {
            "lime_prediction": result["prediction"],
            "lime_confidence": result["confidence"],
            "lime_explanations": result.get("explanations", []),
        }
    except Exception as exc:
        logger.warning("LIME branch failed (non-fatal, model=%s): %s", model_choice, exc, exc_info=True)
        return {"lime_prediction": None, "lime_confidence": None, "lime_explanations": None}


def compute_verdict(
    text_prediction: Optional[str],
    image_prediction: Optional[str],
    clip_reuse: bool,
) -> tuple[str, list[str]]:
    """
    Combine branch results into a cautious overall verdict and reasoning list.

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
