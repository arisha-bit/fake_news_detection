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

    model, tokenizer = load_lstm()

    sequences = tokenizer.texts_to_sequences(
        texts
    )

    padded = pad_sequences(
        sequences,
        maxlen=500
    )

    probabilities = model.predict(
        padded,
        verbose=0
    ).reshape(-1)

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

    # Get explanation for predicted class
    feature_list = explanation.as_list(
        label=predicted_index
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

        # Skip extremely tiny contributions
        if abs(float(impact)) < 0.001:
            continue

        explanations.append({
            "word": clean_word,

            # Keep the SIGNED impact
            "impact": round(
                float(impact),
                4
            ),

            "direction": (
                "REAL"
                if impact > 0
                else "FAKE"
            )
        })

    return {
        "prediction": prediction,

        "confidence": round(
            float(confidence),
            4
        ),

        "explanations": explanations
    }