"""Runs repository — async (FastAPI) and sync (worker) variants.

The two classes mirror each other method-for-method. The async variant uses
AsyncSession; the sync variant uses Session. Both rely on row-level locking
(`with_for_update()`) for status transitions so concurrent instances cannot
clobber each other.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from ..models import Episode, Run


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_id(run_id: str | uuid.UUID) -> uuid.UUID:
    if isinstance(run_id, uuid.UUID):
        return run_id
    return uuid.UUID(run_id)


# ============================================================================
# Sync variant — used by the background worker (pipeline.process_run).
# ============================================================================
class RunsRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        task_name: str,
        uploaded_filename: str,
        model: str,
        run_id: uuid.UUID | None = None,
    ) -> Run:
        run = Run(
            id=run_id or uuid.uuid4(),
            task_name=task_name,
            uploaded_filename=uploaded_filename,
            model=model,
            status="queued",
        )
        self.session.add(run)
        self.session.flush()
        return run

    def get(self, run_id: str | uuid.UUID, *, with_episodes: bool = True) -> Run | None:
        stmt = select(Run).where(Run.id == _coerce_id(run_id))
        if with_episodes:
            stmt = stmt.options(selectinload(Run.episodes))
        return self.session.execute(stmt).scalar_one_or_none()

    def get_for_update(self, run_id: str | uuid.UUID) -> Run | None:
        return (
            self.session.execute(
                select(Run).where(Run.id == _coerce_id(run_id)).with_for_update()
            )
            .scalar_one_or_none()
        )

    def list(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> list[Run]:
        stmt = select(Run).options(selectinload(Run.episodes)).order_by(Run.created_at.desc())
        if status:
            stmt = stmt.where(Run.status == status)
        stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def update_status(
        self,
        run_id: str | uuid.UUID,
        status: str,
        *,
        error_message: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> Run | None:
        run = self.get_for_update(run_id)
        if run is None:
            return None
        run.status = status
        if error_message is not None:
            run.error_message = error_message
        if started_at is not None:
            run.started_at = started_at
        if finished_at is not None:
            run.finished_at = finished_at
        self.session.flush()
        return run

    def set_pause(self, run_id: str | uuid.UUID, requested: bool) -> Run | None:
        run = self.get_for_update(run_id)
        if run is None:
            return None
        run.pause_requested = requested
        if requested:
            if run.status == "running":
                run.status = "pausing"
        else:
            if run.status in ("paused", "pausing"):
                run.status = "queued"
        self.session.flush()
        return run

    def set_artifacts(
        self,
        run_id: str | uuid.UUID,
        *,
        original_zip_uri: str | None = None,
        summary_csv_uri: str | None = None,
        summary_xlsx_uri: str | None = None,
        explorer_html_uri: str | None = None,
    ) -> None:
        run = self.get_for_update(run_id)
        if run is None:
            return
        if original_zip_uri is not None:
            run.original_zip_uri = original_zip_uri
        if summary_csv_uri is not None:
            run.summary_csv_uri = summary_csv_uri
        if summary_xlsx_uri is not None:
            run.summary_xlsx_uri = summary_xlsx_uri
        if explorer_html_uri is not None:
            run.explorer_html_uri = explorer_html_uri
        self.session.flush()

    def recompute_counts(self, run_id: str | uuid.UUID) -> Run | None:
        rid = _coerce_id(run_id)
        run = self.get_for_update(rid)
        if run is None:
            return None
        totals = self.session.execute(
            select(
                func.count().label("total"),
                func.sum(case((Episode.category == "golden", 1), else_=0)).label("golden"),
                func.sum(case((Episode.category == "failure", 1), else_=0)).label("failure"),
            ).where(Episode.run_id == rid)
        ).one()
        run.total_episodes = int(totals.total or 0)
        run.golden_count = int(totals.golden or 0)
        run.failure_count = int(totals.failure or 0)
        self.session.flush()
        return run

    def delete(self, run_id: str | uuid.UUID) -> bool:
        run = self.get(run_id, with_episodes=False)
        if run is None:
            return False
        self.session.delete(run)
        self.session.flush()
        return True

    def claim_finalization(self, run_id: str | uuid.UUID) -> Run | None:
        """Atomically flip status -> 'done' if currently 'running' and no pending episodes.

        Only the winner gets the row back; everyone else gets None and bails on
        writing summaries/explorer. Caller must verify all episodes are terminal.
        """
        rid = _coerce_id(run_id)
        run = self.get_for_update(rid)
        if run is None:
            return None
        if run.status != "running":
            return None
        pending = self.session.execute(
            select(func.count())
            .select_from(Episode)
            .where(Episode.run_id == rid, Episode.status.in_(("pending", "running")))
        ).scalar_one()
        if pending:
            return None
        run.finished_at = _utc_now()
        self.session.flush()
        return run


# ============================================================================
# Async variant — used by FastAPI route handlers.
# ============================================================================
class RunsRepositoryAsync:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        task_name: str,
        uploaded_filename: str,
        model: str,
        run_id: uuid.UUID | None = None,
    ) -> Run:
        run = Run(
            id=run_id or uuid.uuid4(),
            task_name=task_name,
            uploaded_filename=uploaded_filename,
            model=model,
            status="queued",
        )
        self.session.add(run)
        await self.session.flush()
        return run

    async def get(
        self, run_id: str | uuid.UUID, *, with_episodes: bool = True
    ) -> Run | None:
        stmt = select(Run).where(Run.id == _coerce_id(run_id))
        if with_episodes:
            stmt = stmt.options(selectinload(Run.episodes))
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_for_update(self, run_id: str | uuid.UUID) -> Run | None:
        return (
            await self.session.execute(
                select(Run).where(Run.id == _coerce_id(run_id)).with_for_update()
            )
        ).scalar_one_or_none()

    async def list(
        self,
        *,
        limit: int = 100,
        status: str | None = None,
    ) -> list[Run]:
        stmt = (
            select(Run)
            .options(selectinload(Run.episodes))
            .order_by(Run.created_at.desc())
        )
        if status:
            stmt = stmt.where(Run.status == status)
        stmt = stmt.limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def set_pause(self, run_id: str | uuid.UUID, requested: bool) -> Run | None:
        run = await self.get_for_update(run_id)
        if run is None:
            return None
        run.pause_requested = requested
        if requested:
            if run.status == "running":
                run.status = "pausing"
        else:
            if run.status in ("paused", "pausing"):
                run.status = "queued"
        await self.session.flush()
        return run

    async def delete(self, run_id: str | uuid.UUID) -> bool:
        run = await self.get(run_id, with_episodes=False)
        if run is None:
            return False
        await self.session.delete(run)
        await self.session.flush()
        return True

    async def recompute_counts(self, run_id: str | uuid.UUID) -> Run | None:
        rid = _coerce_id(run_id)
        run = await self.get_for_update(rid)
        if run is None:
            return None
        totals = (
            await self.session.execute(
                select(
                    func.count().label("total"),
                    func.sum(case((Episode.category == "golden", 1), else_=0)).label("golden"),
                    func.sum(
                        case((Episode.category == "failure", 1), else_=0)
                    ).label("failure"),
                ).where(Episode.run_id == rid)
            )
        ).one()
        run.total_episodes = int(totals.total or 0)
        run.golden_count = int(totals.golden or 0)
        run.failure_count = int(totals.failure or 0)
        await self.session.flush()
        return run
