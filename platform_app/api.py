"""REST API routes (Postgres-backed).

Response shapes are unchanged from the GCS-only version — see
`serializers.py` for the field-for-field translation from ORM objects to the
old `index.json` / `manifest.json` shapes the frontend expects.
"""

from __future__ import annotations

import io
import logging
import re
import uuid
import zipfile
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from . import pipeline, serializers
from .config import Settings, get_settings
from .db import get_db, sync_session
from .repositories import (
    EpisodesRepository,
    EpisodesRepositoryAsync,
    NewEpisode,
    RunsRepository,
    RunsRepositoryAsync,
)
from .storage import get_storage, key_from_uri, storage_uri_for


log = logging.getLogger("autoeval.api")
router = APIRouter(prefix="/api")


def require_token(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str, *, fallback: str = "episode") -> str:
    """Sanitize a name for use in a Content-Disposition or zip entry path.

    Strips path separators and any non-[A-Za-z0-9._-] runs, collapses to "_".
    """
    cleaned = _FILENAME_SAFE.sub("_", name).strip("._-")
    return cleaned or fallback


def _coerce_run_uuid(run_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="run not found") from e


@router.post("/runs", dependencies=[Depends(require_token)])
async def create_run(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Create a run and kick off the background worker.

    Uses a sync session throughout — the bulk insert + recompute + status
    transitions are easier to reason about transactionally, and we're already
    offloading the zip ingest to a worker thread.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="must upload a .zip file")
    zip_bytes = await file.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="empty upload")

    run_id = uuid.uuid4()
    task_name = file.filename.rsplit(".", 1)[0]
    storage = get_storage(settings)

    # 1. Persist the run as 'queued' so it shows up in /api/runs immediately.
    with sync_session() as s:
        RunsRepository(s).create(
            run_id=run_id,
            task_name=task_name,
            uploaded_filename=file.filename,
            model=settings.openai_model,
        )

    # 2. Unpack the zip (slow; thread pool keeps the event loop free).
    try:
        discovered = await run_in_threadpool(pipeline.ingest_zip, storage, run_id, zip_bytes)
    except Exception as e:  # noqa: BLE001
        log.exception("ingest failed for %s", run_id)
        from datetime import datetime, timezone
        with sync_session() as s:
            RunsRepository(s).update_status(
                run_id,
                "failed",
                error_message=f"ingest failed: {type(e).__name__}: {e}",
                finished_at=datetime.now(timezone.utc),
            )
        raise HTTPException(status_code=400, detail=str(e)) from e

    # 3. Bulk-insert episodes + record artifact URI + recompute counts.
    new_eps: list[NewEpisode] = []
    for ep in discovered.get(pipeline.GOLDEN_CATEGORY, []):
        new_eps.append(
            NewEpisode(
                category=pipeline.GOLDEN_CATEGORY,
                name=ep["name"],
                trajectory_uri=storage_uri_for(storage, ep["trajectory_key"]),
            )
        )
    for ep in discovered.get(pipeline.FAILURE_CATEGORY, []):
        new_eps.append(
            NewEpisode(
                category=pipeline.FAILURE_CATEGORY,
                name=ep["name"],
                trajectory_uri=storage_uri_for(storage, ep["trajectory_key"]),
            )
        )

    with sync_session() as s:
        EpisodesRepository(s).bulk_insert(run_id, new_eps)
        runs_repo = RunsRepository(s)
        runs_repo.set_artifacts(
            run_id,
            original_zip_uri=storage_uri_for(storage, f"runs/{run_id.hex}/original.zip"),
        )
        runs_repo.recompute_counts(run_id)
        run = runs_repo.get(run_id, with_episodes=False)
        # snapshot scalars before the session closes
        status = run.status if run else "queued"
        total = run.total_episodes if run else 0

    background.add_task(pipeline.process_run, run_id)

    return {
        "run_id": run_id.hex,
        "status": status,
        "total_episodes": total,
    }


@router.get("/runs", dependencies=[Depends(require_token)])
async def list_runs(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    repo = RunsRepositoryAsync(db)
    runs = await repo.list(limit=200)
    return [serializers.run_index_entry(r) for r in runs]


@router.get("/runs/{run_id}", dependencies=[Depends(require_token)])
async def get_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rid = _coerce_run_uuid(run_id)
    repo = RunsRepositoryAsync(db)
    run = await repo.get(rid)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return serializers.run_manifest(run)


@router.get("/runs/{run_id}/episodes/{episode_id}", dependencies=[Depends(require_token)])
async def get_episode_result(
    run_id: str,
    episode_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rid = _coerce_run_uuid(run_id)
    ep = await EpisodesRepositoryAsync(db).get(rid, episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="episode not found")
    if not ep.result_uri and ep.result is None:
        return {"status": ep.status, "result": None, "error": ep.error_message or ""}
    if ep.result is not None:
        return {"status": ep.status, "result": ep.result}
    # Backfilled rows may have result_uri but null result — read from storage.
    storage = get_storage(settings)
    return {"status": ep.status, "result": storage.get_json(key_from_uri(ep.result_uri))}


def _download_blob(
    storage_uri: str | None,
    settings: Settings,
    *,
    missing_detail: str,
    media_type: str,
    filename: str,
) -> Response:
    if not storage_uri:
        raise HTTPException(status_code=404, detail=missing_detail)
    storage = get_storage(settings)
    key = key_from_uri(storage_uri)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail=missing_detail)
    return Response(
        content=storage.get_bytes(key),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/runs/{run_id}/golden_summary.csv", dependencies=[Depends(require_token)])
async def download_golden(
    run_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> Response:
    rid = _coerce_run_uuid(run_id)
    run = await RunsRepositoryAsync(db).get(rid, with_episodes=False)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _download_blob(
        run.summary_csv_uri,
        settings,
        missing_detail="golden_summary.csv not yet available",
        media_type="text/csv",
        filename=f"golden_summary_{run_id}.csv",
    )


@router.get("/runs/{run_id}/failure_summary.xlsx", dependencies=[Depends(require_token)])
async def download_failure(
    run_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> Response:
    rid = _coerce_run_uuid(run_id)
    run = await RunsRepositoryAsync(db).get(rid, with_episodes=False)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _download_blob(
        run.summary_xlsx_uri,
        settings,
        missing_detail="failure_summary.xlsx not yet available",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"failure_summary_{run_id}.xlsx",
    )


@router.get(
    "/runs/{run_id}/episodes/{episode_id}/explorer.html",
    dependencies=[Depends(require_token)],
)
async def download_episode_explorer(
    run_id: str,
    episode_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> Response:
    rid = _coerce_run_uuid(run_id)
    ep = await EpisodesRepositoryAsync(db).get(rid, episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="episode not found")
    return _download_blob(
        ep.explorer_html_uri,
        settings,
        missing_detail="explorer.html not yet available",
        media_type="text/html; charset=utf-8",
        filename=f"{_safe_filename(ep.name, fallback=episode_id)}.html",
    )


@router.get(
    "/runs/{run_id}/explorers.zip",
    dependencies=[Depends(require_token)],
)
async def download_explorers_zip(
    run_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Bundle every per-episode explorer.html for this run into a zip.

    Layout inside the zip:  {category}/{episode_name}.html
    """
    rid = _coerce_run_uuid(run_id)
    run = await RunsRepositoryAsync(db).get(rid)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    storage = get_storage(settings)
    seen_names: set[str] = set()
    entries: list[tuple[str, bytes]] = []  # (zip_path, html_bytes)

    for ep in run.episodes:
        if not ep.explorer_html_uri:
            continue
        key = key_from_uri(ep.explorer_html_uri)
        if not key or not storage.exists(key):
            continue
        category = _safe_filename(ep.category, fallback="episode")
        base = _safe_filename(ep.name, fallback=ep.episode_key)
        zip_path = f"{category}/{base}.html"
        # Disambiguate collisions ({category}/{name}.html clashes are unlikely
        # but possible after sanitization; suffix with the episode_key tail).
        if zip_path in seen_names:
            tail = ep.id.hex[:8]
            zip_path = f"{category}/{base}__{tail}.html"
        seen_names.add(zip_path)
        try:
            entries.append((zip_path, storage.get_bytes(key)))
        except Exception:  # noqa: BLE001
            log.exception("could not read explorer for episode %s", ep.episode_key)

    if not entries:
        raise HTTPException(
            status_code=404,
            detail="no per-episode explorer.html files available yet",
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path, data in entries:
            zf.writestr(path, data)

    zip_filename = f"explorers_{_safe_filename(run.task_name, fallback=run_id)}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@router.delete("/runs/{run_id}", dependencies=[Depends(require_token)])
async def delete_run(
    run_id: str,
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rid = _coerce_run_uuid(run_id)
    repo = RunsRepositoryAsync(db)
    deleted = await repo.delete(rid)
    if not deleted:
        raise HTTPException(status_code=404, detail="run not found")
    await db.commit()
    storage = get_storage(settings)
    storage.delete_prefix(f"runs/{rid.hex}")
    return {"deleted": run_id}


@router.post("/runs/{run_id}/pause", dependencies=[Depends(require_token)])
async def pause_run(
    run_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rid = _coerce_run_uuid(run_id)
    repo = RunsRepositoryAsync(db)
    run = await repo.get(rid, with_episodes=False)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status in ("done", "failed"):
        return {"run_id": run_id, "status": run.status, "noop": True}
    run = await repo.set_pause(rid, True)
    assert run is not None
    await db.commit()
    return {"run_id": run_id, "status": run.status}


@router.post("/runs/{run_id}/resume", dependencies=[Depends(require_token)])
async def resume_run(
    run_id: str,
    background: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    rid = _coerce_run_uuid(run_id)
    repo = RunsRepositoryAsync(db)
    run = await repo.get(rid, with_episodes=False)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    if run.status in ("done", "failed"):
        return {"run_id": run_id, "status": run.status, "noop": True}
    run = await repo.set_pause(rid, False)
    assert run is not None
    await db.commit()
    background.add_task(pipeline.process_run, rid)
    return {"run_id": run_id, "status": run.status}
