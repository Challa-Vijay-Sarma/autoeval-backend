"""REST API routes."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from . import manifest, pipeline
from .config import Settings, get_settings
from .storage import get_storage


log = logging.getLogger("autoeval.api")
router = APIRouter(prefix="/api")


def require_token(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None),
) -> None:
    if not settings.api_token:
        return  # auth disabled
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@router.post("/runs", dependencies=[Depends(require_token)])
async def create_run(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="must upload a .zip file")
    zip_bytes = await file.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="empty upload")

    run_id = uuid.uuid4().hex
    task_name = file.filename.rsplit(".", 1)[0]
    storage = get_storage(settings)

    # Build the manifest before unzipping so the row appears in the index immediately.
    m = manifest.new_manifest(run_id, task_name, file.filename, settings.openai_model)
    manifest.write_manifest(storage, m)
    manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))

    # Ingest in a worker thread so the event loop (and /api/health, /api/runs
    # polling) stay responsive while ~500 small bucket writes happen.
    try:
        discovered = await run_in_threadpool(pipeline.ingest_zip, storage, run_id, zip_bytes)
    except Exception as e:  # noqa: BLE001
        log.exception("ingest failed for %s", run_id)
        m["status"] = "failed"
        m["error_message"] = f"ingest failed: {type(e).__name__}: {e}"
        m["finished_at"] = manifest.utc_now_iso()
        manifest.write_manifest(storage, m)
        manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))
        raise HTTPException(status_code=400, detail=str(e)) from e

    episodes: list[dict[str, Any]] = []
    for ep in discovered.get(pipeline.GOLDEN_CATEGORY, []):
        episodes.append(manifest.new_episode(pipeline.GOLDEN_CATEGORY, ep["name"], ep["trajectory_key"]))
    for ep in discovered.get(pipeline.FAILURE_CATEGORY, []):
        episodes.append(manifest.new_episode(pipeline.FAILURE_CATEGORY, ep["name"], ep["trajectory_key"]))

    m["episodes"] = episodes
    m["total_episodes"] = len(episodes)
    m["golden_count"] = sum(1 for ep in episodes if ep["category"] == pipeline.GOLDEN_CATEGORY)
    m["failure_count"] = sum(1 for ep in episodes if ep["category"] == pipeline.FAILURE_CATEGORY)
    manifest.write_manifest(storage, m)
    manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))

    # Kick off background work.
    background.add_task(pipeline.process_run, run_id)
    return {"run_id": run_id, "status": m["status"], "total_episodes": m["total_episodes"]}


@router.get("/runs", dependencies=[Depends(require_token)])
def list_runs(settings: Settings = Depends(get_settings)) -> list[dict[str, Any]]:
    return manifest.read_index(get_storage(settings))


@router.get("/runs/{run_id}", dependencies=[Depends(require_token)])
def get_run(run_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    storage = get_storage(settings)
    if not storage.exists(manifest.manifest_key(run_id)):
        raise HTTPException(status_code=404, detail="run not found")
    return manifest.read_manifest(storage, run_id)


@router.get("/runs/{run_id}/episodes/{episode_id}", dependencies=[Depends(require_token)])
def get_episode_result(
    run_id: str, episode_id: str, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    storage = get_storage(settings)
    m = manifest.read_manifest(storage, run_id)
    ep = next((e for e in m["episodes"] if e["episode_id"] == episode_id), None)
    if not ep:
        raise HTTPException(status_code=404, detail="episode not found")
    if not ep["result_key"]:
        return {"status": ep["status"], "result": None, "error": ep.get("error_message", "")}
    return {"status": ep["status"], "result": storage.get_json(ep["result_key"])}


@router.get("/runs/{run_id}/golden_summary.csv", dependencies=[Depends(require_token)])
def download_golden(run_id: str, settings: Settings = Depends(get_settings)) -> Response:
    storage = get_storage(settings)
    key = f"{manifest.run_prefix(run_id)}/golden_summary.csv"
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="golden_summary.csv not yet available")
    return Response(
        content=storage.get_text(key),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="golden_summary_{run_id}.csv"'},
    )


@router.get("/runs/{run_id}/failure_summary.xlsx", dependencies=[Depends(require_token)])
def download_failure(run_id: str, settings: Settings = Depends(get_settings)) -> Response:
    storage = get_storage(settings)
    key = f"{manifest.run_prefix(run_id)}/failure_summary.xlsx"
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="failure_summary.xlsx not yet available")
    return Response(
        content=storage.get_bytes(key),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="failure_summary_{run_id}.xlsx"'},
    )


@router.delete("/runs/{run_id}", dependencies=[Depends(require_token)])
def delete_run(run_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    storage = get_storage(settings)
    storage.delete_prefix(manifest.run_prefix(run_id))
    manifest.remove_index_entry(storage, run_id)
    return {"deleted": run_id}


@router.post("/runs/{run_id}/pause", dependencies=[Depends(require_token)])
def pause_run(run_id: str, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Signal the worker for this run to stop scheduling new episodes.

    In-flight LLM calls finish (no mid-call cancel). Once they finish, the
    run lands in status='paused' and waits for /resume.
    """
    storage = get_storage(settings)
    if not storage.exists(manifest.manifest_key(run_id)):
        raise HTTPException(status_code=404, detail="run not found")
    m = manifest.read_manifest(storage, run_id)
    if m["status"] in ("done", "failed"):
        return {"run_id": run_id, "status": m["status"], "noop": True}

    pipeline.request_pause(run_id)
    m["pause_requested"] = True
    # If the worker is mid-execution we mark "pausing"; if it hasn't started
    # yet (queued / paused), it stays where it is until process_run reads it.
    if m["status"] == "running":
        m["status"] = "pausing"
    manifest.write_manifest(storage, m)
    manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))
    return {"run_id": run_id, "status": m["status"]}


@router.post("/runs/{run_id}/resume", dependencies=[Depends(require_token)])
def resume_run(
    run_id: str,
    background: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Clear the pause flag and re-schedule the worker."""
    storage = get_storage(settings)
    if not storage.exists(manifest.manifest_key(run_id)):
        raise HTTPException(status_code=404, detail="run not found")
    m = manifest.read_manifest(storage, run_id)
    if m["status"] in ("done", "failed"):
        return {"run_id": run_id, "status": m["status"], "noop": True}

    pipeline.request_resume(run_id)
    m["pause_requested"] = False
    if m["status"] in ("paused", "pausing"):
        m["status"] = "queued"
    manifest.write_manifest(storage, m)
    manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))

    background.add_task(pipeline.process_run, run_id)
    return {"run_id": run_id, "status": m["status"]}
