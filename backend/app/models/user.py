import uuid

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import UUID

from app.db.database import Base

class User(Base):

    __tablename__="users"

    id=Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    username=Column(
        String,
        nullable=False
    )

    email=Column(
        String,
        unique=True,
        nullable=False
    )

    password_hash=Column(
        String,
        nullable=False
    )

    role=Column(
        String,
        default="user"
    )