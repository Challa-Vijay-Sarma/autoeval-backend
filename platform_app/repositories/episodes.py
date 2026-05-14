"""Episodes repository.

The critical operation is `claim_pending`: it uses `FOR UPDATE SKIP LOCKED` so
two workers (in the same process or different Cloud Run instances) can never
pick up the same episode.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..models import Episode


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_id(value: str | uuid.UUID) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(value)


@dataclass
class NewEpisode:
    category: str        # 'golden' | 'failure'
    name: str
    trajectory_uri: str  # gs:// or storage-relative key


# ============================================================================
# Sync variant
# ============================================================================
class EpisodesRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def bulk_insert(self, run_id: str | uuid.UUID, items: list[NewEpisode]) -> list[Episode]:
        rid = _coerce_id(run_id)
        rows = [
            Episode(
                run_id=rid,
                episode_key=f"{it.category}__{it.name}",
                category=it.category,
                name=it.name,
                trajectory_uri=it.trajectory_uri,
                status="pending",
            )
            for it in items
        ]
        if not rows:
            return []
        self.session.add_all(rows)
        self.session.flush()
        return rows

    def list_for_run(self, run_id: str | uuid.UUID) -> list[Episode]:
        rid = _coerce_id(run_id)
        return list(
            self.session.execute(
                select(Episode).where(Episode.run_id == rid).order_by(Episode.created_at)
            )
            .scalars()
            .all()
        )

    def get(self, run_id: str | uuid.UUID, episode_key: str) -> Episode | None:
        rid = _coerce_id(run_id)
        return (
            self.session.execute(
                select(Episode).where(
                    Episode.run_id == rid, Episode.episode_key == episode_key
                )
            )
            .scalar_one_or_none()
        )

    def claim_pending(self, run_id: str | uuid.UUID, *, limit: int = 1) -> list[Episode]:
        """Pick up to `limit` pending episodes and flip them to 'running'.

        Uses FOR UPDATE SKIP LOCKED so concurrent workers (across instances)
        get disjoint sets.
        """
        rid = _coerce_id(run_id)
        candidates = (
            self.session.execute(
                select(Episode)
                .where(Episode.run_id == rid, Episode.status == "pending")
                .order_by(Episode.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )
        now = _utc_now()
        claimed: list[Episode] = []
        for ep in candidates:
            ep.status = "running"
            ep.started_at = now
            ep.error_message = ""
            claimed.append(ep)
        if claimed:
            self.session.flush()
        return claimed

    def mark_done(
        self,
        episode_id: uuid.UUID,
        *,
        result: dict[str, Any] | None,
        summary: dict[str, Any],
        result_uri: str | None,
    ) -> Episode | None:
        ep = (
            self.session.execute(
                select(Episode).where(Episode.id == episode_id).with_for_update()
            )
            .scalar_one_or_none()
        )
        if ep is None or ep.status != "running":
            return ep
        ep.status = "done"
        ep.finished_at = _utc_now()
        ep.result = result
        ep.summary = summary
        if result_uri is not None:
            ep.result_uri = result_uri
        ep.error_message = ""
        self.session.flush()
        return ep

    def mark_error(self, episode_id: uuid.UUID, error_message: str) -> Episode | None:
        ep = (
            self.session.execute(
                select(Episode).where(Episode.id == episode_id).with_for_update()
            )
            .scalar_one_or_none()
        )
        if ep is None or ep.status != "running":
            return ep
        ep.status = "error"
        ep.finished_at = _utc_now()
        ep.error_message = error_message
        self.session.flush()
        return ep

    def set_explorer_uri(self, episode_id: uuid.UUID, uri: str) -> None:
        ep = (
            self.session.execute(
                select(Episode).where(Episode.id == episode_id).with_for_update()
            )
            .scalar_one_or_none()
        )
        if ep is None:
            return
        ep.explorer_html_uri = uri
        self.session.flush()

    def reset_stale_running(
        self, run_id: str | uuid.UUID, *, stale_after_seconds: int = 900
    ) -> int:
        """Reset any 'running' episode older than the threshold back to 'pending'.

        Called at the start of `process_run` to recover from worker crashes.
        Returns the number of rows reset.
        """
        rid = _coerce_id(run_id)
        cutoff = _utc_now() - timedelta(seconds=stale_after_seconds)
        stale = (
            self.session.execute(
                select(Episode)
                .where(
                    Episode.run_id == rid,
                    Episode.status == "running",
                    (Episode.started_at.is_(None)) | (Episode.started_at < cutoff),
                )
                .with_for_update()
            )
            .scalars()
            .all()
        )
        for ep in stale:
            ep.status = "pending"
            ep.started_at = None
            if not ep.error_message:
                ep.error_message = "reset after worker restart"
        if stale:
            self.session.flush()
        return len(stale)

    def count_remaining(self, run_id: str | uuid.UUID) -> int:
        """Episodes still pending or running for this run."""
        rid = _coerce_id(run_id)
        from sqlalchemy import func

        return int(
            self.session.execute(
                select(func.count())
                .select_from(Episode)
                .where(Episode.run_id == rid, Episode.status.in_(("pending", "running")))
            ).scalar_one()
        )


# ============================================================================
# Async variant (read-only paths from FastAPI)
# ============================================================================
class EpisodesRepositoryAsync:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_for_run(self, run_id: str | uuid.UUID) -> list[Episode]:
        rid = _coerce_id(run_id)
        return list(
            (
                await self.session.execute(
                    select(Episode)
                    .where(Episode.run_id == rid)
                    .order_by(Episode.created_at)
                )
            )
            .scalars()
            .all()
        )

    async def get(self, run_id: str | uuid.UUID, episode_key: str) -> Episode | None:
        rid = _coerce_id(run_id)
        return (
            await self.session.execute(
                select(Episode).where(
                    Episode.run_id == rid, Episode.episode_key == episode_key
                )
            )
        ).scalar_one_or_none()
