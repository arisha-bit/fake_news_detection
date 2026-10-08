"""
Pydantic schemas for multimodal image verification.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class LIMEFeature(BaseModel):
    """A single LIME feature contribution."""
    word: str
    impact: float
    direction: str  # "REAL" or "FAKE"


class ImageXAIResult(BaseModel):
    """XAI results for image prediction."""
    # Grad-CAM
    heatmap_url: Optional[str] = Field(
        None,
        description="URL path to the Grad-CAM heatmap PNG, served from /static/heatmaps/"
    )
    gradcam_prediction: Optional[str] = None
    gradcam_confidence: Optional[float] = None

    # LIME (for lstm/bert text models when OCR text is available)
    lime_prediction: Optional[str] = None
    lime_confidence: Optional[float] = None
    lime_explanations: Optional[List[LIMEFeature]] = None


class ImageClassificationResponse(BaseModel):
    """Response from POST /predict/image — pure image classification."""
    prediction: str = Field(..., description="FAKE or REAL")
    confidence: float
    class_probabilities: dict
    heatmap_url: Optional[str] = Field(
        None,
        description="URL path to Grad-CAM heatmap PNG, or null if generation failed"
    )


class SimilarArticle(BaseModel):
    """A single CLIP-based similar article match."""
    rank: int
    title: str
    label: str
    similarity: float
    potential_source: str = ""
    date: str = ""
    snippet: str = ""


class ImageVerificationResponse(BaseModel):
    """
    Full multimodal verification report from POST /verify/image.
    Combines OCR text prediction + image classification + CLIP similarity + XAI.
    """
    # Text (OCR) branch
    ocr_text: str = Field(..., description="Text extracted from image via OCR")
    text_prediction: str = Field(..., description="FAKE or REAL based on OCR text")
    text_confidence: float

    # Image classification branch
    image_prediction: str = Field(..., description="FAKE or REAL based on image pixels")
    image_confidence: float
    image_class_probabilities: dict

    # CLIP reverse search branch
    similar_articles: list[SimilarArticle]
    clip_reuse_detected: bool

    # XAI branch (Grad-CAM + LIME)
    xai: Optional[ImageXAIResult] = Field(
        None,
        description="Explainability results: Grad-CAM heatmap URL and LIME text explanations"
    )

    # Combined verdict
    overall_verdict: str = Field(
        ...,
        description="LIKELY FAKE | LIKELY REAL | UNCERTAIN | LIKELY MISLEADING"
    )
    reasoning: list[str]

    # DB references
    prediction_id: Optional[UUID] = None
    upload_id: Optional[UUID] = None


class ImageVerificationRecord(BaseModel):
    """Schema for stored verification records."""
    id: UUID
    user_id: UUID
    filename: str
    ocr_text: str
    text_prediction: str
    text_confidence: float
    image_prediction: str
    image_confidence: float
    overall_verdict: str
    created_at: datetime

    class Config:
        from_attributes = True
