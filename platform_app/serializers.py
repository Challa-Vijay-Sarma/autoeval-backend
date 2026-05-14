"""ORM -> JSON serializers that preserve the exact shape the frontend expects.

These match what `manifest.py` produced before the Postgres migration:
  - index_entry_from_manifest(): one row in the /api/runs list response
  - manifest dict: full /api/runs/{id} response

`trajectory_key` and `result_key` are kept as bucket-relative keys (not gs://
URIs) so the frontend's download/explorer links keep working unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import Episode, Run
from .storage import key_from_uri


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_index_entry(run: Run) -> dict[str, Any]:
    """Matches the old index.json row shape."""
    return {
        "run_id": run.id.hex,
        "task_name": run.task_name,
        "uploaded_filename": run.uploaded_filename,
        "model": run.model,
        "status": run.status,
        "pause_requested": run.pause_requested,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "total_episodes": run.total_episodes,
        "golden_count": run.golden_count,
        "failure_count": run.failure_count,
    }


def episode_dict(ep: Episode) -> dict[str, Any]:
    """Matches the old per-episode shape inside manifest.json, plus has_explorer."""
    return {
        "episode_id": ep.episode_key,
        "category": ep.category,
        "name": ep.name,
        "trajectory_key": key_from_uri(ep.trajectory_uri),
        "result_key": key_from_uri(ep.result_uri) if ep.result_uri else "",
        "status": ep.status,
        "started_at": _iso(ep.started_at),
        "finished_at": _iso(ep.finished_at),
        "error_message": ep.error_message or "",
        "summary": ep.summary or {},
        "has_explorer": bool(ep.explorer_html_uri),
    }


def run_manifest(run: Run, episodes: list[Episode] | None = None) -> dict[str, Any]:
    """Matches the old manifest.json shape (run-level fields + episodes array)."""
    eps = episodes if episodes is not None else list(run.episodes)
    return {
        "run_id": run.id.hex,
        "task_name": run.task_name,
        "uploaded_filename": run.uploaded_filename,
        "model": run.model,
        "status": run.status,
        "pause_requested": run.pause_requested,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
        "total_episodes": run.total_episodes,
        "golden_count": run.golden_count,
        "failure_count": run.failure_count,
        "error_message": run.error_message or "",
        "episodes": [episode_dict(ep) for ep in eps],
    }
