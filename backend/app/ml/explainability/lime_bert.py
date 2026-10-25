"""
LIME explainability for the DistilBERT model.

Uses LimeTextExplainer with a predict_proba wrapper around the
DistilBertForSequenceClassification model via load_bert().

Class order: [0=FAKE, 1=REAL] — matches model training.
"""

import logging
import re

import numpy as np
from lime.lime_text import LimeTextExplainer

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

_explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])


def predict_proba_for_lime(texts: list) -> "np.ndarray":
    """
    Probability function for LIME — returns (n_samples, 2) array.
    Column 0 = FAKE probability, Column 1 = REAL probability.
    """
    import torch
    from app.ml.inference.predict_bert import load_bert

    tokenizer, model = load_bert()

    # Validate input: ensure all texts are non-empty strings
    texts_list = [str(t).strip() for t in texts]
    if not texts_list or all(len(t) == 0 for t in texts_list):
        logger.warning("Empty text list passed to predict_proba_for_lime")
        # Return neutral probabilities (0.5, 0.5) for all empty inputs
        return np.array([[0.5, 0.5]] * len(texts_list))

    inputs = tokenizer(
        texts_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Validate tokenizer output
    if inputs.input_ids.shape[0] == 0 or inputs.input_ids.shape[1] == 0:
        logger.warning("Tokenizer produced empty tensor for LIME prediction")
        return np.array([[0.5, 0.5]] * len(texts_list))

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    result = probs.cpu().numpy()
    # Validate output shape
    if result.shape[0] != len(texts_list):
        logger.warning(
            "Output shape mismatch: %d predictions for %d texts",
            result.shape[0],
            len(texts_list),
        )
    return result


def explain_bert(text: str, num_features: int = 10) -> dict:
    """
    Generate LIME explanation for the DistilBERT model.

    Returns:
        {prediction, confidence, explanations, fake_features, real_features}
    """
    # Truncate to 512 words to keep LIME sampling time manageable
    words = text.split()
    text_for_lime = " ".join(words[:512]) if len(words) > 512 else text

    explanation = _explainer.explain_instance(
        text_for_lime,
        predict_proba_for_lime,
        num_features=num_features,
        labels=[0, 1],
    )

    probabilities = predict_proba_for_lime([text_for_lime])[0]
    predicted_index = int(np.argmax(probabilities))
    prediction = "REAL" if predicted_index == 1 else "FAKE"
    confidence = float(probabilities[predicted_index])

    # Get explanation for REAL class (label=1):
    # positive weight pushes toward REAL (class 1)
    # negative weight pushes toward FAKE (class 0)
    feature_list = explanation.as_list(label=1)

    explanations = []
    fake_features = []
    real_features = []

    for word, impact in feature_list:
        word = str(word).strip()
        clean_word = re.sub(r"[^\w'-]", "", word).strip()
        if not clean_word or clean_word.lower() in STOPWORDS:
            continue
        imp_val = float(impact)
        if not np.isfinite(imp_val) or abs(imp_val) < 0.0001:
            continue

        direction = "REAL" if imp_val > 0 else "FAKE"
        if direction == "REAL":
            real_features.append(clean_word)
        else:
            fake_features.append(clean_word)

        explanations.append({
            "word": clean_word,
            "impact": round(abs(imp_val), 4),
            "weight": round(imp_val, 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "output_space": "probability (ΔP)",
        "explanations": explanations[:num_features],
        "fake_features": fake_features,
        "real_features": real_features,
    }
