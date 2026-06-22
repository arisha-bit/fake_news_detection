import uuid

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Float

from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base
class ModelMetric(Base):

    __tablename__ = "model_metrics"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    model_name = Column(String)

    accuracy = Column(Float)

    precision = Column(Float)

    recall = Column(Float)

    f1_score = Column(Float)