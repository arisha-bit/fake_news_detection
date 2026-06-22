"""initial migration

Revision ID: ab78fa56dcfe
Revises: 
Create Date: 2026-06-21 23:37:24.156607

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'ab78fa56dcfe'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('email', sa.String(), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('role', sa.String(), default='user'),
    )

    op.create_table(
        'predictions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('prediction', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    op.create_table(
        'feedback',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('prediction_id', UUID(as_uuid=True), sa.ForeignKey('predictions.id'), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('feedback')
    op.drop_table('predictions')
    op.drop_table('users')
