"""
SHAP explainability for DistilBERT using shap.Explainer with a HuggingFace pipeline.

Uses the PARTITION explainer via shap.Explainer(pipeline(...)) on CPU.
Synchronized with threading.Lock to avoid HuggingFace fast tokenizer
Rust PyO3 `RuntimeError: Already borrowed` under concurrent requests.
"""

import logging
import os
import re
import threading
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

_explainer = None
_bert_shap_lock = threading.Lock()


def _get_explainer():
    global _explainer
    if _explainer is None:
        with _bert_shap_lock:
            if _explainer is None:
                import shap
                import torch
                from transformers import pipeline as hf_pipeline
                from app.ml.inference.predict_bert import load_bert

                tokenizer, model = load_bert()

                # Ensure model is on CPU
                if torch.cuda.is_available():
                    model = model.to(torch.device("cpu"))
                    logger.info("Moved DistilBERT model to CPU explicitly")

                # Build HuggingFace text-classification pipeline; device=-1 forces CPU
                pipe = hf_pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    top_k=None,   # return all labels
                    device=-1,
                    truncation=True,
                    max_length=512,
                )

                # shap.Explainer auto-selects PartitionExplainer for pipelines
                _explainer = shap.Explainer(pipe)
                logger.info("DistilBERT SHAP explainer ready on CPU.")
    return _explainer


def explain_bert_shap(text: str, top_k: int = 10) -> dict:
    """
    Generate SHAP explanation for the DistilBERT model.

    Returns:
        {
            prediction,
            confidence,
            output_space: "probability (ΔP)",
            explanations: [{word, impact, weight, direction}],
            fake_features,
            real_features
        }
    """
    from app.ml.inference.predict_bert import predict_news_bert

    if not text or not text.strip():
        return {
            "prediction": "FAKE",
            "confidence": 0.5,
            "output_space": "probability (ΔP)",
            "explanations": [],
            "fake_features": [],
            "real_features": [],
        }

    # Get the actual model prediction first
    pred_res = predict_news_bert(text)
    prediction = pred_res["prediction"]
    confidence = pred_res["confidence"]

    # Keep text manageable on CPU without losing context
    words = text.split()
    text_for_shap = " ".join(words[:300]) if len(words) > 300 else text

    # Protect entire SHAP partition evaluation with lock to prevent
    # Hugging Face fast tokenizer `RuntimeError: Already borrowed`
    with _bert_shap_lock:
        explainer = _get_explainer()
        shap_values = explainer([text_for_shap])

    # shap_values.data[0]   — list of tokens/words
    # shap_values.values[0] — shape (n_tokens, n_classes)
    tokens = shap_values.data[0]
    values = shap_values.values[0]

    # Validate token-value alignment
    if len(tokens) != len(values):
        logger.warning(
            "Token-value mismatch in BERT SHAP: %d tokens vs %d values; truncating to shorter",
            len(tokens),
            len(values),
        )
        min_len = min(len(tokens), len(values))
        tokens = tokens[:min_len]
        values = values[:min_len]

    explanations = []
    fake_features = []
    real_features = []

    for i, token in enumerate(tokens):
        word = str(token).strip()
        clean_word = re.sub(r"[^\w'-]", "", word).strip()
        if not clean_word or clean_word.lower() in STOPWORDS:
            continue

        # Bounds check
        if i >= len(values):
            logger.warning("Value index %d out of range in BERT SHAP; skipping", i)
            break

        # index 1 = REAL, index 0 = FAKE (matches DistilBERT label order)
        if values.ndim == 2:
            val = float(values[i, 1])
        else:
            val = float(values[i])

        if not np.isfinite(val) or abs(val) < 0.0001:
            continue

        direction = "REAL" if val > 0 else "FAKE"
        if direction == "REAL":
            real_features.append(clean_word)
        else:
            fake_features.append(clean_word)

        explanations.append({
            "word": clean_word,
            "impact": round(abs(val), 4),
            "weight": round(val, 4),
            "direction": direction,
        })

    explanations.sort(key=lambda x: x["impact"], reverse=True)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "output_space": "probability (ΔP)",
        "explanations": explanations[:top_k],
        "fake_features": fake_features[:top_k],
        "real_features": real_features[:top_k],
    }
