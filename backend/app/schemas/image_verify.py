"""
Pydantic schemas for multimodal image verification (Parts 2, 5, 6).
"""

from typing import Optional
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class ImageClassificationResponse(BaseModel):
    """Response from POST /predict/image — pure image classification."""
    prediction: str = Field(..., description="FAKE or REAL")
    confidence: float
    class_probabilities: dict[str, float]


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
    Combines OCR text prediction + image classification + CLIP similarity.
    """
    # Text (OCR) branch
    ocr_text: str = Field(..., description="Text extracted from image via OCR")
    text_prediction: str = Field(..., description="FAKE or REAL based on OCR text")
    text_confidence: float

    # Image classification branch
    image_prediction: str = Field(..., description="FAKE or REAL based on image pixels")
    image_confidence: float
    image_class_probabilities: dict[str, float]

    # CLIP reverse search branch
    similar_articles: list[SimilarArticle]
    clip_reuse_detected: bool

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
