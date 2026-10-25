import numpy as np
import re

from lime.lime_text import LimeTextExplainer

from app.ml.inference.predict_lstm import load_lstm

from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Stopwords to hide from the explanation
# --------------------------------------------------

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


# --------------------------------------------------
# LIME explainer
# --------------------------------------------------

explainer = LimeTextExplainer(
    class_names=["FAKE", "REAL"]
)


# --------------------------------------------------
# Prediction function for LIME
# --------------------------------------------------

def predict_proba_for_lime(texts):
    import logging
    
    logger = logging.getLogger(__name__)
    model, tokenizer = load_lstm()

    # Ensure texts are strings
    texts = [str(t).strip() for t in texts]
    
    # Check for empty inputs
    if not texts or all(len(t) == 0 for t in texts):
        logger.warning("Empty text list passed to LSTM LIME prediction")
        # Return neutral probabilities
        return np.array([[0.5, 0.5]] * len(texts))

    sequences = tokenizer.texts_to_sequences(texts)
    
    # Check for empty sequences (texts not in vocabulary)
    if not sequences or all(len(seq) == 0 for seq in sequences):
        logger.warning("Tokenizer produced empty sequences for LSTM LIME")
        # Return neutral probabilities
        return np.array([[0.5, 0.5]] * len(texts))

    padded = pad_sequences(
        sequences,
        maxlen=500
    )

    # Validate padded shape
    if padded.size == 0:
        logger.warning("Padded sequences are empty")
        return np.array([[0.5, 0.5]] * len(texts))

    probabilities = model.predict(
        padded,
        verbose=0
    ).reshape(-1)
    
    # Validate probabilities
    if len(probabilities) != len(texts):
        logger.warning(
            "Probability count mismatch: %d predictions for %d texts",
            len(probabilities),
            len(texts),
        )
    
    # Handle NaN or invalid probabilities
    probabilities = np.nan_to_num(probabilities, nan=0.5, posinf=0.999, neginf=0.001)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    # Column 0 = FAKE
    # Column 1 = REAL

    return np.column_stack([
        1 - probabilities,
        probabilities
    ])


# --------------------------------------------------
# Explain one LSTM prediction
# --------------------------------------------------

def explain_lstm(
    text: str,
    num_features: int = 10
):

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        text,
        predict_proba_for_lime,
        num_features=num_features,
        labels=[0, 1]
    )

    # Get model probabilities
    probabilities = predict_proba_for_lime(
        [text]
    )[0]

    # Determine prediction
    predicted_index = int(
        np.argmax(probabilities)
    )

    prediction = (
        "REAL"
        if predicted_index == 1
        else "FAKE"
    )

    confidence = probabilities[
        predicted_index
    ]

    # Get explanation for REAL class (label=1):
    # positive weight pushes toward REAL (class 1)
    # negative weight pushes toward FAKE (class 0)
    feature_list = explanation.as_list(
        label=1
    )

    explanations = []

    for word, impact in feature_list:

        # Convert NumPy string → normal Python string
        word = str(word).strip()

        # Remove unnecessary punctuation
        clean_word = re.sub(
            r"[^a-zA-Z0-9'-]",
            "",
            word
        )

        # Skip empty values
        if not clean_word:
            continue

        # Skip common stopwords
        if clean_word.lower() in STOPWORDS:
            continue

        imp_val = float(impact)
        if not np.isfinite(imp_val) or abs(imp_val) < 0.0001:
            continue

        direction = (
            "REAL"
            if imp_val > 0
            else "FAKE"
        )

        explanations.append({
            "word": clean_word,
            "impact": round(abs(imp_val), 4),
            "weight": round(imp_val, 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)
    fake_features = [e["word"] for e in explanations if e["direction"] == "FAKE"]
    real_features = [e["word"] for e in explanations if e["direction"] == "REAL"]

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "output_space": "probability (ΔP)",
        "explanations": explanations[:num_features],
        "fake_features": fake_features,
        "real_features": real_features,
    }
