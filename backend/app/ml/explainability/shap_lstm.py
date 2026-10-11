"""
SHAP explainability for the LSTM model.

Uses shap.Explainer with a Text masker (PartitionExplainer).
The prediction function wraps the Keras LSTM's tokenize-pad-predict pipeline.

To keep latency acceptable on CPU, text is truncated to 300 words
before SHAP sampling (the LSTM itself still pads to 500 tokens).
"""

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "am", "be", "been", "being", "to", "of", "in",
    "on", "for", "from", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its",
    "as", "at", "by", "about", "into", "than",
    "then", "there", "their", "they", "them",
    "he", "she", "his", "her", "we", "you", "your",
    "i", "me", "my",
}

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        import shap
        from app.ml.inference.predict_lstm import load_lstm

        # Pre-load the model and tokenizer so the closure captures them
        model, tokenizer = load_lstm()

        def _predict_proba(texts):
            from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa

            texts_list = [str(t) for t in texts]
            seqs = tokenizer.texts_to_sequences(texts_list)
            padded = pad_sequences(seqs, maxlen=500)
            preds = model.predict(padded, verbose=0).reshape(-1)
            # Column 0 = FAKE, Column 1 = REAL
            return np.column_stack([1 - preds, preds])

        masker = shap.maskers.Text(r"\W+")
        _explainer = shap.Explainer(_predict_proba, masker)
        logger.info("LSTM SHAP explainer ready.")
    return _explainer


def explain_lstm_shap(text: str, top_k: int = 10) -> dict:
    """
    Generate SHAP explanation for the LSTM model.

    Returns:
        {prediction, confidence, explanations, fake_features, real_features}
    """
    from app.ml.inference.predict_lstm import load_lstm
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa

    # Get prediction from the actual model
    model, tokenizer = load_lstm()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=500)
    prob_real = float(model.predict(padded, verbose=0).reshape(-1)[0])
    prediction = "REAL" if prob_real >= 0.5 else "FAKE"
    confidence = prob_real if prediction == "REAL" else (1.0 - prob_real)

    # Truncate to 300 words to keep SHAP sampling manageable on CPU
    words = text.split()
    text_for_shap = " ".join(words[:300]) if len(words) > 300 else text

    try:
        explainer = _get_explainer()
        shap_values = explainer([text_for_shap])
    except Exception as exc:
        logger.warning("LSTM SHAP failed: %s", exc, exc_info=True)
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "explanations": [],
            "fake_features": [],
            "real_features": [],
        }

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    # Validate token-value alignment
    if len(tokens) != len(values):
        logger.warning(
            "Token-value mismatch: %d tokens vs %d values; truncating to shorter",
            len(tokens),
            len(values),
        )
        min_len = min(len(tokens), len(values))
        tokens = tokens[:min_len]
        values = values[:min_len]

    explanations = []
    fake_features = []
    real_features = []

    for i, token in enumerate(tokens):
        word = str(token).strip()
        clean_word = re.sub(r"[^\w'-]", "", word).strip()
        if not clean_word or clean_word.lower() in STOPWORDS:
            continue

        # Bounds check
        if i >= len(values):
            logger.warning("Value index %d out of range; skipping", i)
            break

        # Index 1 = REAL (positive → REAL, negative → FAKE)
        val = float(values[i, 1]) if values.ndim == 2 else float(values[i])
        if abs(val) < 0.0001:
            continue

        direction = "REAL" if val > 0 else "FAKE"
        if direction == "REAL":
            real_features.append(clean_word)
        else:
            fake_features.append(clean_word)

        explanations.append({
            "word": clean_word,
            "impact": round(abs(val), 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "explanations": explanations[:top_k],
        "fake_features": fake_features[:top_k],
        "real_features": real_features[:top_k],
    }
