"""add feedback table

Revision ID: 3f059415b3ae
Revises: ab78fa56dcfe
Create Date: 2026-06-22 10:56:30.246276

"""
from typing import Sequence, Union

from alembic import op


revision: str = '3f059415b3ae'
down_revision: Union[str, Sequence[str], None] = 'ab78fa56dcfe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass  # already handled in initial migration


def downgrade() -> None:
    pass
