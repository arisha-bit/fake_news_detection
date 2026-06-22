import uuid

from sqlalchemy import Column
from sqlalchemy import Boolean
from sqlalchemy import String
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base


class Feedback(Base):

    __tablename__ = "feedback"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    prediction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id"),
        nullable=False
    )

    model_name = Column(
        String,
        nullable=False
    )

    is_correct = Column(
        Boolean,
        nullable=False
    )