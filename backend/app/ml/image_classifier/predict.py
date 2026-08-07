"""
Image Classification Inference — EfficientNetB0

Singleton model loader with prediction function.
Returns prediction label, confidence, and class probabilities.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import HTTPException

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "saved_models" / "image_classifier.pt"

# Inference transform (no augmentation)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Singletons
_model: Optional[nn.Module] = None
_class_names: Optional[list[str]] = None
_device: Optional[torch.device] = None


def _get_model():
    """Lazy-load the trained EfficientNetB0 model once."""
    global _model, _class_names, _device

    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Image classifier model not found. "
                    "Run: python -m app.ml.image_classifier.train --data_dir <path> "
                    "to train the model first."
                ),
            )
        try:
            _device = torch.device("cpu")  # CPU-only in Docker
            checkpoint = torch.load(MODEL_PATH, map_location=_device)
            _class_names = checkpoint["class_names"]
            num_classes = len(_class_names)

            _model = models.efficientnet_b0(weights=None)
            in_features = _model.classifier[1].in_features
            _model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, num_classes),
            )
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.eval()
            _model.to(_device)

            logger.info(
                "Image classifier loaded. Classes: %s, val_acc=%.4f",
                _class_names,
                checkpoint.get("val_acc", 0),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Failed to load image classifier: %s", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Image classifier failed to load: {str(exc)}",
            ) from exc

    return _model, _class_names, _device


def predict_image(image_path: str) -> dict:
    """
    Run the trained EfficientNetB0 classifier on an image.

    Args:
        image_path: Path to image file on disk.

    Returns:
        {
            "prediction": "FAKE" or "REAL",
            "confidence": float 0.0-1.0,
            "class_probabilities": {"FAKE": 0.92, "REAL": 0.08}
        }

    Raises:
        HTTP 404 — image file not found.
        HTTP 422 — cannot open image.
        HTTP 503 — model not trained yet.
        HTTP 500 — inference failure.
    """
    from pathlib import Path as _Path
    from PIL import Image

    path = _Path(image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot open image: {str(exc)}",
        ) from exc

    model, class_names, device = _get_model()

    try:
        tensor = _TRANSFORM(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx].upper()
        confidence = float(round(probs[pred_idx], 4))

        class_probs = {
            name.upper(): float(round(float(p), 4))
            for name, p in zip(class_names, probs)
        }

        logger.info(
            "Image classification: %s (%.2f%%) — %s",
            pred_label, confidence * 100, image_path
        )

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "class_probabilities": class_probs,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Image inference error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Image classification failed: {str(exc)}",
        ) from exc
