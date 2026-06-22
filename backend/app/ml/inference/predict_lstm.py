from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "saved_models" /
    "lstm.keras"
)

TOKENIZER_PATH = (
    BASE_DIR /
    "saved_models" /
    "lstm_tokenizer.pkl"
)

model = None
tokenizer = None


def load_lstm():

    global model
    global tokenizer

    if model is None:
        model = load_model(
            MODEL_PATH
        )

    if tokenizer is None:
        tokenizer = joblib.load(
            TOKENIZER_PATH
        )

    return model, tokenizer


def predict_news_lstm(
    text: str
):

    model, tokenizer = load_lstm()

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=500
    )

    probability = model.predict(
        padded,
        verbose=0
    )[0][0]

    prediction = (
        "REAL"
        if probability >= 0.5
        else "FAKE"
    )

    return {
        "prediction": prediction,
        "confidence": float(
            round(
                max(
                    probability,
                    1 - probability
                ),
                4
            )
        )
    }