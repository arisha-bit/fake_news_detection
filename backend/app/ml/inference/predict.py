import joblib
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

# Load model and vectorizer once when the application starts
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
model = joblib.load(
    MODEL_PATH
)

vectorizer = joblib.load(
    VECTORIZER_PATH
)


def predict_news(text: str):
    # Convert text into TF-IDF features
    vector = vectorizer.transform([text])

    # Predict class
    prediction = model.predict(vector)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(vector)[0]

    # Confidence of the predicted class
    confidence = probabilities[prediction]

    return {
    "prediction":
        "REAL"
        if prediction == 1
        else "FAKE",

    "confidence":
        float(
            round(
                confidence,
                4
            )
        )
}