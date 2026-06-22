from pathlib import Path

import os
import torch

# Force CPU — no CUDA in Docker
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.device("cpu")

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "saved_models" /
    "distilbert"
)

tokenizer = None
model = None


def load_bert():

    global tokenizer
    global model

    if tokenizer is None:

        tokenizer = (
            DistilBertTokenizerFast
            .from_pretrained(
                MODEL_PATH
            )
        )

    if model is None:

        model = (
            DistilBertForSequenceClassification
            .from_pretrained(
                MODEL_PATH
            )
        )

        model.eval()

    return tokenizer, model


def predict_news_bert(
    text: str
):

    tokenizer, model = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence = torch.max(
            probabilities
        ).item()

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    prediction = (
        "REAL"
        if predicted_class == 1
        else "FAKE"
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }