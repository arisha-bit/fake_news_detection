"""
ResNet18 image classification inference.

Model is loaded lazily on first call — NOT at module import time.
This prevents crashes when best_image_model.pth is absent at startup.

Class mapping:
    0 = FAKE
    1 = REAL
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "saved_models"
    / "best_image_model.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["FAKE", "REAL"]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_model: Optional[nn.Module] = None


def _get_model() -> nn.Module:
    """Load and cache the ResNet18 model on first call."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ResNet18 model weights not found at {MODEL_PATH}. "
                "Ensure best_image_model.pth is present in app/ml/saved_models/."
            )
        logger.info("Loading ResNet18 from %s", MODEL_PATH)
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
        m = m.to(DEVICE)
        m.eval()
        _model = m
        logger.info("ResNet18 model ready.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_image(image_path: str) -> dict:
    """
    Classify an image as FAKE or REAL using the trained ResNet18 model.

    Args:
        image_path: Path to the image file.

    Returns:
        {
            "prediction": "FAKE" | "REAL",
            "confidence": float  (0–100, percentage)
        }

    Raises:
        FileNotFoundError — model weights missing.
        Exception — image read / inference failure.
    """
    model = _get_model()

    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        prediction = CLASS_NAMES[predicted_index]
        confidence = probabilities[0, predicted_index].item()

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
    }
