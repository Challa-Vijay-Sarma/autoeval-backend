"""episode explorer_html_uri

Revision ID: 0002_episode_explorer_uri
Revises: 0001_initial
Create Date: 2026-05-14
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0002_episode_explorer_uri"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "episodes",
        sa.Column("explorer_html_uri", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("episodes", "explorer_html_uri")
