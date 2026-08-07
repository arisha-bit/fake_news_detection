"""add image_verifications table

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-07 00:00:00.000000
"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_verifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("ocr_text", sa.Text(), nullable=True),
        sa.Column("text_prediction", sa.String(), nullable=True),
        sa.Column("text_confidence", sa.Float(), nullable=True),
        sa.Column("image_prediction", sa.String(), nullable=True),
        sa.Column("image_confidence", sa.Float(), nullable=True),
        sa.Column("clip_reuse_detected", sa.Boolean(), default=False),
        sa.Column("overall_verdict", sa.String(), nullable=True),
        sa.Column("model_used", sa.String(), default="logistic"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("image_verifications")
