"""Per-run manifest.json + global index.json read/write helpers.

The bucket is the database. Index.json is a flat array; one entry per run.
Manifest.json is the authoritative state of one run, including every episode.
All writes are atomic via the storage backend's put_json (which uses tmp+rename
on local, and replace-on-write on GCS).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from .storage import StorageBackend


INDEX_KEY = "index.json"
_index_lock = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------
# Index (list of all runs)
# ----------------------------------------------------------------------------
def read_index(storage: StorageBackend) -> list[dict[str, Any]]:
    if not storage.exists(INDEX_KEY):
        return []
    data = storage.get_json(INDEX_KEY)
    if not isinstance(data, list):
        return []
    return data


def write_index(storage: StorageBackend, runs: list[dict[str, Any]]) -> None:
    storage.put_json(INDEX_KEY, runs)


def upsert_index_entry(storage: StorageBackend, entry: dict[str, Any]) -> None:
    """Insert or replace the entry for entry['run_id']. Newest first."""
    with _index_lock:
        runs = read_index(storage)
        runs = [r for r in runs if r.get("run_id") != entry["run_id"]]
        runs.insert(0, entry)
        write_index(storage, runs)


def remove_index_entry(storage: StorageBackend, run_id: str) -> None:
    with _index_lock:
        runs = read_index(storage)
        runs = [r for r in runs if r.get("run_id") != run_id]
        write_index(storage, runs)


# ----------------------------------------------------------------------------
# Per-run manifest
# ----------------------------------------------------------------------------
def run_prefix(run_id: str) -> str:
    return f"runs/{run_id}"


def manifest_key(run_id: str) -> str:
    return f"{run_prefix(run_id)}/manifest.json"


def new_manifest(
    run_id: str,
    task_name: str,
    uploaded_filename: str,
    model: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "task_name": task_name,
        "uploaded_filename": uploaded_filename,
        "model": model,
        "status": "queued",
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "total_episodes": 0,
        "golden_count": 0,
        "failure_count": 0,
        "error_message": "",
        "episodes": [],  # list of episode dicts (see new_episode below)
    }


def new_episode(
    category: str,
    name: str,
    trajectory_key: str,
) -> dict[str, Any]:
    return {
        "episode_id": f"{category}__{name}",
        "category": category,  # "golden" | "failure"
        "name": name,
        "trajectory_key": trajectory_key,
        "result_key": "",
        "status": "pending",  # pending | running | done | error
        "started_at": None,
        "finished_at": None,
        "error_message": "",
        "summary": {},  # the row that ends up in CSV/XLSX (or compact UI row)
    }


def read_manifest(storage: StorageBackend, run_id: str) -> dict[str, Any]:
    return storage.get_json(manifest_key(run_id))  # type: ignore[return-value]


def write_manifest(storage: StorageBackend, manifest: dict[str, Any]) -> None:
    storage.put_json(manifest_key(manifest["run_id"]), manifest)


def index_entry_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "task_name": manifest["task_name"],
        "uploaded_filename": manifest["uploaded_filename"],
        "model": manifest["model"],
        "status": manifest["status"],
        "created_at": manifest["created_at"],
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "total_episodes": manifest["total_episodes"],
        "golden_count": manifest["golden_count"],
        "failure_count": manifest["failure_count"],
    }
