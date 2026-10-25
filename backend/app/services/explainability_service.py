import yake

# Initialize the YAKE keyword extractor
keyword_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    top=5
)

def extract_keywords(text: str) -> list[str]:
    """Extracts top keywords from the text using YAKE."""
    keywords = keyword_extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]


# Predefined list of sensational/clickbait phrases
CLICKBAIT_PHRASES = [
    "shocking", "miracle", "secret", "you won't believe", 
    "breaking", "unbelievable", "instant", "guaranteed", 
    "exposed", "truth revealed"
]

def clickbait_score(text: str) -> int:
    """Calculates a score (0-100) based on the presence of clickbait phrases."""
    text = text.lower()
    matches = 0

    for phrase in CLICKBAIT_PHRASES:
        if phrase in text:
            matches += 1

    return min(100, matches * 20)


def generate_explanation(prediction: str, score: int) -> str:
    """Generates a user-friendly explanation based on model prediction and clickbait score."""
    if prediction == "FAKE":
        if score > 50:
            return (
                "The article contains sensational language and "
                "clickbait patterns often associated with fake news."
            )
        else:
            return (
                "The model detected linguistic patterns commonly "
                "found in misleading news articles."
            )
    else:
        # Handles "REAL" news or any other prediction outcome
        return (
            "The article uses neutral language "
            "and resembles authentic news reporting."
        )


def generate_model_explanation(text: str, model_choice: str = "logistic", method: str = "shap", top_k: int = 10) -> dict:
    """
    Generate SHAP or LIME explanation for any of the three models:
    - logistic
    - lstm
    - bert
    """
    model_choice = model_choice.lower()
    method = method.lower()

    if model_choice not in ("logistic", "lstm", "bert"):
        raise ValueError(f"Unsupported model: {model_choice}. Choose logistic, lstm, or bert.")

    if method not in ("shap", "lime"):
        raise ValueError(f"Unsupported method: {method}. Choose shap or lime.")

    if model_choice == "logistic":
        if method == "shap":
            from app.ml.explainability.shap_explainer import explain_prediction
            res = explain_prediction(text, top_k=top_k)
        else:
            from app.ml.explainability.lime_logistic import explain_logistic_lime
            res = explain_logistic_lime(text, num_features=top_k)

    elif model_choice == "lstm":
        if method == "shap":
            from app.ml.explainability.shap_lstm import explain_lstm_shap
            res = explain_lstm_shap(text, top_k=top_k)
        else:
            from app.ml.explainability.lime_lstm import explain_lstm
            res = explain_lstm(text, num_features=top_k)

    elif model_choice == "bert":
        if method == "shap":
            from app.ml.explainability.shap_bert import explain_bert_shap
            res = explain_bert_shap(text, top_k=top_k)
        else:
            from app.ml.explainability.lime_bert import explain_bert
            res = explain_bert(text, num_features=top_k)

    return {
        "model": model_choice,
        "method": method,
        "explanation_type": method,  # "shap" or "lime"
        "prediction": res.get("prediction", "UNKNOWN"),
        "confidence": res.get("confidence", 0.0),
        "output_space": res.get("output_space", "log-odds" if (model_choice == "logistic" and method == "shap") else "probability (ΔP)"),
        "explanations": res.get("explanations", []),
        "fake_features": res.get("fake_features", []),
        "real_features": res.get("real_features", []),
    }
