"""
LIME explainability for Logistic Regression + TF-IDF model.
"""

import logging
from pathlib import Path
import re
import numpy as np
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "logistic.pkl"
VECTORIZER_PATH = BASE_DIR / "saved_models" / "tfidf.pkl"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "to", "of", "in",
    "on", "for", "from", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its",
    "as", "at", "by", "about", "into", "than",
    "then", "there", "their", "they", "them",
    "he", "she", "his", "her", "we", "you", "your",
    "i", "me", "my"
}

_model = None
_vectorizer = None
_explainer = None


def _get_components():
    global _model, _vectorizer, _explainer
    if _model is None:
        import joblib
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Logistic model not found at {MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found at {VECTORIZER_PATH}")
        _model = joblib.load(MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    return _model, _vectorizer, _explainer


def predict_proba_for_lime(texts):
    import scipy.sparse
    
    model, vectorizer, _ = _get_components()
    vectors = vectorizer.transform(texts)
    
    # Validate sparse matrix is not empty
    if scipy.sparse.issparse(vectors):
        if vectors.nnz == 0:  # nnz = number of non-zero elements
            logger.warning("Vectorizer produced empty sparse matrix for LIME")
            # Return neutral probabilities for all texts
            return np.array([[0.5, 0.5]] * vectors.shape[0])
    else:
        # Dense array case
        if vectors.size == 0:
            logger.warning("Vectorizer produced empty array for LIME")
            return np.array([[0.5, 0.5]] * vectors.shape[0])
    
    probabilities = model.predict_proba(vectors)
    # Ensure output is 2D (n_samples, 2)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    if probabilities.shape[1] != 2:
        logger.warning(
            "Unexpected proba shape: %s; padding to (n, 2)",
            probabilities.shape,
        )
        if probabilities.shape[1] == 1:
            # Single column, add inverse
            probabilities = np.column_stack([1 - probabilities, probabilities])
    
    return probabilities


def explain_logistic_lime(text: str, num_features: int = 10) -> dict:
    """
    Generate LIME explanation for Logistic Regression.
    """
    model, vectorizer, explainer = _get_components()

    # Probabilities for this instance
    probabilities = predict_proba_for_lime([text])[0]
    predicted_index = int(np.argmax(probabilities))
    prediction = "REAL" if predicted_index == 1 else "FAKE"
    confidence = float(probabilities[predicted_index])

    explanation = explainer.explain_instance(
        text,
        predict_proba_for_lime,
        num_features=num_features,
        labels=[0, 1]
    )

    feature_list = explanation.as_list(label=1)  # label 1 = REAL, positive pushes REAL, negative pushes FAKE

    explanations = []
    fake_features = []
    real_features = []

    for word, impact in feature_list:
        clean_word = re.sub(r"[^\w'-]", "", str(word)).strip()
        if not clean_word or clean_word.lower() in STOPWORDS:
            continue
        if abs(float(impact)) < 0.0001:
            continue

        direction = "REAL" if impact > 0 else "FAKE"
        if direction == "REAL":
            real_features.append(clean_word)
        else:
            fake_features.append(clean_word)

        explanations.append({
            "word": clean_word,
            "impact": round(abs(float(impact)), 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanations": explanations[:num_features],
        "fake_features": fake_features,
        "real_features": real_features,
    }
