"""SQLAlchemy 2.x models for runs + episodes.

Mirrors the shape of the old GCS manifest.json + index.json but normalized
into two tables so we can use row-level locking and indexed queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


RUN_STATUSES = ("queued", "running", "pausing", "paused", "done", "failed")
EPISODE_CATEGORIES = ("golden", "failure")
EPISODE_STATUSES = ("pending", "running", "done", "error")

RunStatusEnum = Enum(*RUN_STATUSES, name="run_status", create_type=False)
EpisodeCategoryEnum = Enum(*EPISODE_CATEGORIES, name="episode_category", create_type=False)
EpisodeStatusEnum = Enum(*EPISODE_STATUSES, name="episode_status", create_type=False)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    task_name: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_filename: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(
        RunStatusEnum, nullable=False, server_default=text("'queued'")
    )
    pause_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )

    total_episodes: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    golden_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    error_message: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))

    original_zip_uri: Mapped[str | None] = mapped_column(Text)
    summary_csv_uri: Mapped[str | None] = mapped_column(Text)
    summary_xlsx_uri: Mapped[str | None] = mapped_column(Text)
    explorer_html_uri: Mapped[str | None] = mapped_column(Text)

    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )

    episodes: Mapped[list["Episode"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="Episode.created_at",
        lazy="selectin",
    )

    __table_args__ = (
        Index("runs_status_idx", "status"),
        Index("runs_created_at_desc_idx", "created_at"),
        Index("runs_status_created_at_idx", "status", "created_at"),
    )


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    episode_key: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(EpisodeCategoryEnum, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)

    trajectory_uri: Mapped[str] = mapped_column(Text, nullable=False)
    result_uri: Mapped[str | None] = mapped_column(Text)
    explorer_html_uri: Mapped[str | None] = mapped_column(Text)

    status: Mapped[str] = mapped_column(
        EpisodeStatusEnum, nullable=False, server_default=text("'pending'")
    )
    error_message: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))

    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    summary: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )

    run: Mapped[Run] = relationship(back_populates="episodes")

    __table_args__ = (
        UniqueConstraint("run_id", "episode_key", name="episodes_run_key_uniq"),
        Index("episodes_run_status_idx", "run_id", "status"),
        Index("episodes_run_category_idx", "run_id", "category"),
        Index("episodes_status_idx", "status"),
    )
