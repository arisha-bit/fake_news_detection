"""
SQLAlchemy model for uploaded image files.
Tracks file metadata and links to predictions.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class UploadedFile(Base):
    """Stores metadata for every file uploaded by a user."""

    __tablename__ = "uploaded_files"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )

    # Nullable — only set after a successful prediction
    prediction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id"),
        nullable=True
    )

    # UUID-based safe filename stored on disk
    filename = Column(String, nullable=False)

    # Original filename as uploaded by the user
    original_filename = Column(String, nullable=False)

    # MIME type e.g. image/jpeg
    file_type = Column(String, nullable=False)

    # Absolute or relative path to the saved file
    file_path = Column(String, nullable=False)

    uploaded_at = Column(
        DateTime,
        default=datetime.utcnow
    )
