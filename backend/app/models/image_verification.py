"""
SQLAlchemy model for multimodal image verification records.
Stores both OCR text branch and image classification branch results.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, Text, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class ImageVerification(Base):
    """Full image verification record combining OCR + image classifier + CLIP."""

    __tablename__ = "image_verifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(UUID(as_uuid=True), nullable=True)

    # File info
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    # OCR branch
    ocr_text = Column(Text, nullable=True)
    text_prediction = Column(String, nullable=True)
    text_confidence = Column(Float, nullable=True)

    # Image classification branch
    image_prediction = Column(String, nullable=True)
    image_confidence = Column(Float, nullable=True)

    # CLIP branch
    clip_reuse_detected = Column(Boolean, default=False)

    # Combined result
    overall_verdict = Column(String, nullable=True)

    model_used = Column(String, default="logistic")

    created_at = Column(DateTime, default=datetime.utcnow)
