"""
DistilBERT inference module.

Model was saved with transformers==5.12.1 — we use that same version.
Loads lazily on first call. Always runs on CPU.

Class mapping (matches training):
    0 = FAKE
    1 = REAL

The torch.accelerator issue in transformers 5.x on CPU-only machines is
avoided by:
  1. Setting CUDA_VISIBLE_DEVICES=-1 before any import so torch sees no GPU.
  2. NOT passing device_map to from_pretrained (avoids accelerate dispatch).
  3. Explicitly calling model.to("cpu") after loading.
"""

import os
import logging
from pathlib import Path

# Must be set before torch or transformers are imported
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# Prevent transformers from trying to use accelerate device dispatch
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "distilbert"

_tokenizer = None
_model = None


def load_bert():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            DistilBertTokenizerFast,
            DistilBertForSequenceClassification,
        )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"DistilBERT model directory not found at {MODEL_PATH}. "
                "Ensure saved_models/distilbert/ contains config.json, "
                "model.safetensors, tokenizer.json, tokenizer_config.json."
            )

        logger.info("Loading DistilBERT tokenizer from %s", MODEL_PATH)
        _tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))

        logger.info("Loading DistilBERT model weights from %s", MODEL_PATH)
        # Do NOT pass device_map here — avoids accelerate dispatch on CPU-only
        _model = DistilBertForSequenceClassification.from_pretrained(
            str(MODEL_PATH),
            low_cpu_mem_usage=False,   # avoids meta-device init that needs accelerator
        )
        _model = _model.to(torch.device("cpu"))
        _model.eval()
        logger.info("DistilBERT ready on CPU.")

    return _tokenizer, _model


def predict_news_bert(text: str) -> dict:
    """
    Classify *text* as FAKE or REAL using the fine-tuned DistilBERT model.

    Returns:
        {"prediction": "FAKE"|"REAL", "confidence": float}
    """
    import torch  # noqa: PLC0415

    tokenizer, model = load_bert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    # Ensure all inputs are on CPU
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

    prediction = "REAL" if predicted_class == 1 else "FAKE"
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
