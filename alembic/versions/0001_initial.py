"""initial schema: runs + episodes

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-14
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


RUN_STATUSES = ("queued", "running", "pausing", "paused", "done", "failed")
EPISODE_CATEGORIES = ("golden", "failure")
EPISODE_STATUSES = ("pending", "running", "done", "error")


def upgrade() -> None:
    # pgcrypto provides gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    run_status = postgresql.ENUM(*RUN_STATUSES, name="run_status", create_type=True)
    episode_category = postgresql.ENUM(
        *EPISODE_CATEGORIES, name="episode_category", create_type=True
    )
    episode_status = postgresql.ENUM(
        *EPISODE_STATUSES, name="episode_status", create_type=True
    )
    run_status.create(op.get_bind(), checkfirst=True)
    episode_category.create(op.get_bind(), checkfirst=True)
    episode_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("task_name", sa.Text(), nullable=False),
        sa.Column("uploaded_filename", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(*RUN_STATUSES, name="run_status", create_type=False),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column(
            "pause_requested",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column("total_episodes", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("golden_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("original_zip_uri", sa.Text(), nullable=True),
        sa.Column("summary_csv_uri", sa.Text(), nullable=True),
        sa.Column("summary_xlsx_uri", sa.Text(), nullable=True),
        sa.Column("explorer_html_uri", sa.Text(), nullable=True),
        sa.Column(
            "extra",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("runs_status_idx", "runs", ["status"])
    op.create_index("runs_created_at_desc_idx", "runs", [sa.text("created_at DESC")])
    op.create_index(
        "runs_status_created_at_idx", "runs", ["status", sa.text("created_at DESC")]
    )

    op.create_table(
        "episodes",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("episode_key", sa.Text(), nullable=False),
        sa.Column(
            "category",
            postgresql.ENUM(
                *EPISODE_CATEGORIES, name="episode_category", create_type=False
            ),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("trajectory_uri", sa.Text(), nullable=False),
        sa.Column("result_uri", sa.Text(), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(*EPISODE_STATUSES, name="episode_status", create_type=False),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("error_message", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column(
            "summary",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("run_id", "episode_key", name="episodes_run_key_uniq"),
    )
    op.create_index("episodes_run_status_idx", "episodes", ["run_id", "status"])
    op.create_index("episodes_run_category_idx", "episodes", ["run_id", "category"])
    op.create_index("episodes_status_idx", "episodes", ["status"])

    # updated_at triggers — keep them in lockstep with row mutations.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
          NEW.updated_at = now();
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER runs_set_updated_at
        BEFORE UPDATE ON runs
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """
    )
    op.execute(
        """
        CREATE TRIGGER episodes_set_updated_at
        BEFORE UPDATE ON episodes
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS episodes_set_updated_at ON episodes")
    op.execute("DROP TRIGGER IF EXISTS runs_set_updated_at ON runs")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")
    op.drop_index("episodes_status_idx", table_name="episodes")
    op.drop_index("episodes_run_category_idx", table_name="episodes")
    op.drop_index("episodes_run_status_idx", table_name="episodes")
    op.drop_table("episodes")
    op.drop_index("runs_status_created_at_idx", table_name="runs")
    op.drop_index("runs_created_at_desc_idx", table_name="runs")
    op.drop_index("runs_status_idx", table_name="runs")
    op.drop_table("runs")
    op.execute("DROP TYPE IF EXISTS episode_status")
    op.execute("DROP TYPE IF EXISTS episode_category")
    op.execute("DROP TYPE IF EXISTS run_status")
