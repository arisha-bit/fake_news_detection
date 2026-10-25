"""
Pydantic schemas for the image upload feature.
Keeps request/response contracts separate from ORM models.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """
    Returned after a successful image upload + prediction.
    Includes image classification, Grad-CAM heatmap, OCR text, 
    text prediction, and explainability data (keywords, clickbait, XAI).
    """

    # Image classification
    image_prediction: str  # "FAKE" | "REAL"
    image_confidence: float
    
    # Grad-CAM explainability (non-fatal if unavailable)
    heatmap_url: Optional[str] = None
    
    # OCR extraction
    extracted_text: str
    has_meaningful_text: bool = False

    # Text prediction (only when OCR has >20 meaningful characters)
    text_prediction: Optional[str] = None
    text_confidence: Optional[float] = None
    
    # Text explainability
    keywords: list[str]
    clickbait_score: float
    explanation: str
    
    # XAI (model explainability) — LIME or SHAP
    xai: Optional[dict] = None
    
    # Database references
    prediction_id: Optional[UUID] = None
    uploaded_file_id: UUID


class UploadedFileResponse(BaseModel):
    """
    Returned when querying a stored uploaded file record.
    """

    id: UUID
    user_id: UUID
    prediction_id: Optional[UUID]
    filename: str
    original_filename: str
    file_type: str
    file_path: str
    uploaded_at: datetime

    class Config:
        from_attributes = True
