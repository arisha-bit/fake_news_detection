"""
Grad-CAM explainability for ResNet18 image classification.

Reuses the shared ResNet18 singleton from predict_image.py.
Does NOT load a second copy of the model weights.

Target layer: model.layer4[-1]
  - The last BasicBlock of ResNet18's 4th residual stage.
  - Produces 7×7 spatial feature maps — the standard and correct
    Grad-CAM target for ResNet18.
"""

import logging
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — must match predict_image.py
# ---------------------------------------------------------------------------

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


def _get_model():
    """Delegate to the shared singleton in predict_image — no double load."""
    from app.ml.inference.predict_image import _get_model as _shared_get_model
    return _shared_get_model()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam(image_path: str, output_path: str) -> dict:
    """
    Generate a Grad-CAM heatmap overlay for the given image.

    Uses the shared ResNet18 singleton (loaded once, reused here).
    Targets model.layer4[-1] — correct for ResNet18.

    Args:
        image_path:  Path to the input image file.
        output_path: Path where the heatmap overlay PNG is saved.
                     Parent directory must exist.

    Returns:
        {
            "prediction": "FAKE" | "REAL",
            "confidence": float (0–1),
            "heatmap_path": output_path
        }

    Raises:
        FileNotFoundError — model weights missing.
        ImportError       — pytorch_grad_cam not installed.
        Exception         — image read / inference failure.
    """
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    t0 = time.perf_counter()
    model = _get_model()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))
    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Forward pass for prediction (no grad needed here)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    # Grad-CAM — runs a second forward+backward with gradients enabled
    # target_layer = model.layer4[-1] is the standard ResNet18 Grad-CAM target
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Overlay heatmap on the resized RGB image
    rgb_image = numpy_from_pil(image_resized)
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    # Save PNG
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(output_path)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Grad-CAM generated in %.3fs: %s → %s (prediction=%s, conf=%.4f)",
        elapsed, image_path, output_path, CLASS_NAMES[predicted_class], confidence,
    )

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4),
        "heatmap_path": output_path,
    }


def numpy_from_pil(pil_image) -> "np.ndarray":
    """Convert PIL image to float32 numpy array in [0, 1] range."""
    import numpy as np
    return np.array(pil_image).astype(np.float32) / 255.0
