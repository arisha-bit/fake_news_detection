"""
Grad-CAM explainability for ResNet18 image classification.

Model is loaded lazily on first call — not at module import time.
This prevents crashes when best_image_model.pth is missing at startup.
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
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "saved_models" / "best_image_model.pth"

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
    """Load and cache the ResNet18 model. Raises on missing weights."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ResNet18 weights not found at {MODEL_PATH}. "
                "Ensure best_image_model.pth is present."
            )
        logger.info("Loading ResNet18 for Grad-CAM from %s", MODEL_PATH)
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
        m = m.to(DEVICE)
        m.eval()
        _model = m
        logger.info("ResNet18 Grad-CAM model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam(image_path: str, output_path: str) -> dict:
    """
    Generate a Grad-CAM heatmap overlay for the given image.

    Args:
        image_path:  Path to the input image file.
        output_path: Path where the heatmap overlay PNG is saved.

    Returns:
        {
            "prediction": "FAKE" | "REAL",
            "confidence": float (0–1),
            "heatmap_path": output_path
        }

    Raises:
        FileNotFoundError — model weights missing.
        Exception — image read / inference failure.
    """
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = _get_model()

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))

    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Forward pass for prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # Grad-CAM — must run with grad enabled
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay heatmap on resized image
    rgb_image = np.array(image_resized).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    Image.fromarray(visualization).save(output_path)

    logger.info(
        "Grad-CAM generated: %s → %s (prediction=%s, conf=%.4f)",
        image_path, output_path, CLASS_NAMES[predicted_class], confidence,
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4),
        "heatmap_path": output_path,
    }
