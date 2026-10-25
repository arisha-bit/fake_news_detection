"""
SHAP explainability for Logistic Regression + TF-IDF.

Models are loaded lazily on first call to avoid startup crashes
if saved_models are missing.

IMPORTANT: SHAP explains the Logistic Regression model ONLY.
It does not explain LSTM or DistilBERT predictions.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "saved_models" / "logistic.pkl"
VECTORIZER_PATH = BASE_DIR / "saved_models" / "tfidf.pkl"

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_model = None
_vectorizer = None
_explainer = None


def _get_components():
    """Load model, vectorizer, and SHAP explainer on first call."""
    global _model, _vectorizer, _explainer

    if _model is None:
        import joblib
        import shap

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Logistic model not found at {MODEL_PATH}"
            )
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {VECTORIZER_PATH}"
            )

        logger.info("Loading logistic.pkl and tfidf.pkl for SHAP...")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)

        _explainer = shap.LinearExplainer(
            _model,
            _vectorizer.transform(["This is a background example for SHAP."])
        )
        logger.info("SHAP explainer ready.")

    return _model, _vectorizer, _explainer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_prediction(text: str, top_k: int = 10) -> dict:
    """
    Generate a SHAP explanation for the Logistic Regression prediction.

    This function explains the LOGISTIC REGRESSION model only.
    Do not use this to explain LSTM or DistilBERT predictions.

    Args:
        text:   The article text to explain.
        top_k:  Max number of influential words to return.

    Returns:
        {
            "prediction": "REAL" | "FAKE",
            "explanations": [
                {"word": str, "impact": float, "direction": "REAL"|"FAKE"},
                ...
            ]
        }

    Raises:
        FileNotFoundError — model files missing.
        Exception — inference failure.
    """
    model, vectorizer, explainer = _get_components()

    # TF-IDF transform
    vector = vectorizer.transform([text])

    # Logistic Regression prediction
    prediction_int = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(probabilities[prediction_int])

    # SHAP values
    shap_values = explainer(vector)
    values = shap_values.values

    # Handle both 2D and 3D SHAP output shapes
    if values.ndim == 3:
        # Shape (1, n_features, n_classes) — take class 1 (REAL)
        values = values[0, :, 1]
    else:
        # Shape (1, n_features)
        values = values[0]

    feature_names = vectorizer.get_feature_names_out()

    # Only include words that actually appear in this text
    feature_indices = vector.nonzero()[1]

    explanations = []
    fake_features = []
    real_features = []
    for idx in feature_indices:
        # Bounds check: ensure idx is within SHAP values range
        if idx < 0 or idx >= len(values):
            logger.warning(
                "Feature index %d out of range [0, %d); skipping",
                idx,
                len(values),
            )
            continue
        
        impact = float(values[idx])
        if not np.isfinite(impact) or abs(impact) < 0.0001:
            continue

        # Positive SHAP → pushes toward REAL (class 1)
        # Negative SHAP → pushes toward FAKE (class 0)
        direction = "REAL" if impact > 0 else "FAKE"
        word_str = str(feature_names[idx])
        if direction == "REAL":
            real_features.append(word_str)
        else:
            fake_features.append(word_str)

        explanations.append({
            "word": word_str,   # ensure plain Python str, not np.str_
            "impact": round(abs(impact), 4),
            "weight": round(impact, 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)

    logger.debug(
        "SHAP explanation: prediction=%s, top_features=%s",
        "REAL" if prediction_int == 1 else "FAKE",
        [e["word"] for e in explanations[:5]],
    )

    return {
        "prediction": "REAL" if prediction_int == 1 else "FAKE",
        "confidence": round(confidence, 4),
        "output_space": "log-odds (margin)",
        "explanations": explanations[:top_k],
        "fake_features": fake_features,
        "real_features": real_features,
    }

