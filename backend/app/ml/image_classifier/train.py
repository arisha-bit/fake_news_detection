"""
Image Classifier Training Pipeline — EfficientNetB0

Uses transfer learning on EfficientNetB0 (ImageNet pretrained) for binary
fake/real image classification.

Compatible datasets:
  - Fakeddit (r/photoshopbattles vs r/news images)
  - CASIA image tampering dataset
  - Any binary folder structure: dataset/train/real/, dataset/train/fake/

Usage:
    python -m app.ml.image_classifier.train \
        --data_dir datasets/image_dataset \
        --epochs 10 \
        --batch_size 32

Output:
    app/ml/saved_models/image_classifier.pt
    app/ml/saved_models/image_classifier_metrics.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_PATH = SAVE_DIR / "image_classifier.pt"
METRICS_PATH = SAVE_DIR / "image_classifier_metrics.json"

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2) -> nn.Module:
    """
    EfficientNetB0 with a custom binary classification head.
    All backbone layers frozen initially — only the classifier is trained.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
):
    """
    Full training loop with validation, early stopping, and metric export.
    """
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # Dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=TRAIN_TRANSFORMS)
    class_names = full_dataset.classes
    logger.info("Classes: %s", class_names)
    logger.info("Total images: %d", len(full_dataset))

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Use val transforms for validation split
    val_ds.dataset.transform = VAL_TRANSFORMS

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step()

        logger.info(
            "Epoch %d/%d — loss=%.4f  val_acc=%.4f",
            epoch + 1, epochs, running_loss / len(train_loader), val_acc
        )
        history.append({"epoch": epoch + 1, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
            }, MODEL_PATH)
            logger.info("  ✓ Saved best model (val_acc=%.4f)", val_acc)

    # Final evaluation
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "precision": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "class_names": class_names,
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names),
        "best_val_acc": best_val_acc,
        "history": history,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete. Metrics saved to %s", METRICS_PATH)
    logger.info("Final accuracy: %.4f", metrics["accuracy"])
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image fake/real classifier")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (ImageFolder structure)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr)
