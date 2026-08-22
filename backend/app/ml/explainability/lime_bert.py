import numpy as np

from lime.lime_text import LimeTextExplainer

from app.ml.inference.predict_bert import load_bert


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

    tokenizer, model = load_bert()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    model.eval()

    import torch

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

    return probabilities.cpu().numpy()


# --------------------------------------------------
# Explain one DistilBERT prediction
# --------------------------------------------------

def explain_bert(
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

        # Convert to normal Python string
        word = str(word).strip()

        # Remove common stopwords
        if word.lower() in STOPWORDS:
            continue

        # Ignore extremely tiny contributions
        if abs(float(impact)) < 0.001:
            continue

        explanations.append({
            "word": word,

            # Keep the signed impact
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