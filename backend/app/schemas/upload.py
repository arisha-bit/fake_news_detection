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
    Mirrors PredictionResponse but includes OCR-specific fields.
    """

    prediction: str
    confidence: float
    extracted_text: str
    keywords: list[str]
    clickbait_score: int
    explanation: str
    prediction_id: UUID
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
