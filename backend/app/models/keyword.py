import uuid

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base
class Keyword(Base):

    __tablename__ = "keywords"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    prediction_id = Column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id")
    )

    keyword = Column(String)

    score = Column(Float)
