"""add feedback table

Revision ID: 912d82530b37
Revises: 3f059415b3ae
Create Date: 2026-06-22 11:09:25.867220

"""
from typing import Sequence, Union

from alembic import op


revision: str = '912d82530b37'
down_revision: Union[str, Sequence[str], None] = '3f059415b3ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass  # already handled in initial migration


def downgrade() -> None:
    pass
