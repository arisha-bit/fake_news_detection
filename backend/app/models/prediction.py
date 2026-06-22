import uuid

from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    Text,
    ForeignKey,
    DateTime
)

from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id")
    )

    text = Column(Text)

    prediction = Column(String)

    confidence = Column(Float)

    model_name = Column(String)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )