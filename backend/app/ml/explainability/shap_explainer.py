import joblib
import shap
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "saved_models"
    / "logistic.pkl"
)

VECTORIZER_PATH = (
    BASE_DIR
    / "saved_models"
    / "tfidf.pkl"
)


# --------------------------------------------------
# Load model and TF-IDF vectorizer
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------

explainer = shap.LinearExplainer(
    model,
    vectorizer.transform([
        "This is a background example for SHAP."
    ])
)


# --------------------------------------------------
# Explain prediction
# --------------------------------------------------


def explain_prediction(text: str, top_k: int = 10):

    # Convert text into TF-IDF features
    vector = vectorizer.transform([text])

    # Get the model prediction
    prediction = model.predict(vector)[0]

    # Get SHAP explanation
    shap_values = explainer(vector)

    values = shap_values.values

    # Handle SHAP output shape
    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    # TF-IDF feature names
    feature_names = (
        vectorizer.get_feature_names_out()
    )

    # Only consider words actually present in the article
    feature_indices = vector.nonzero()[1]

    explanations = []

    for index in feature_indices:

        impact = float(values[index])

        # For our binary Logistic Regression:
        #
        # Positive SHAP → REAL (class 1)
        # Negative SHAP → FAKE (class 0)

        if impact > 0:
            direction = "REAL"
        else:
            direction = "FAKE"

        explanations.append({
            "word": feature_names[index],
            "impact": round(
                abs(impact),
                4
            ),
            "direction": direction
        })

    # Most influential features first
    explanations.sort(
        key=lambda x: x["impact"],
        reverse=True
    )

    return {
        "prediction": (
            "REAL"
            if prediction == 1
            else "FAKE"
        ),
        "explanations": explanations[:top_k]
    }