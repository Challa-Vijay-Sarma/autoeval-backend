"""Zip ingestion + episode discovery + background worker.

The worker is the heart of the platform. It runs in-process via FastAPI's
BackgroundTasks. On Cloud Run with --min-instances=1 --no-cpu-throttling,
the instance stays warm and the worker survives between requests.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from . import evaluators, manifest
from .config import get_settings
from .storage import StorageBackend, get_storage


log = logging.getLogger("autoeval.pipeline")


GOLDEN_CATEGORY = "golden"
FAILURE_CATEGORY = "failure"
GOLDEN_DIRS = {"Golden_trajectories", "golden_trajectories"}
FAILURE_DIRS = {"Failure_trajectories", "failure_trajectories"}


# ----------------------------------------------------------------------------
# Pause / resume primitives
# ----------------------------------------------------------------------------
# In-process pause flags. Survives until uvicorn restarts; the persistent
# pause_requested flag on the manifest is the source of truth across restarts.
_pause_events: dict[str, threading.Event] = {}
_pause_events_lock = threading.Lock()

# Per-run lock so two `process_run` invocations for the same run can't race
# (e.g. an immediate resume click while the prior worker is still tearing down).
_run_locks: dict[str, threading.Lock] = {}
_run_locks_lock = threading.Lock()


def _get_pause_event(run_id: str) -> threading.Event:
    with _pause_events_lock:
        ev = _pause_events.get(run_id)
        if ev is None:
            ev = threading.Event()
            _pause_events[run_id] = ev
        return ev


def _get_run_lock(run_id: str) -> threading.Lock:
    with _run_locks_lock:
        lock = _run_locks.get(run_id)
        if lock is None:
            lock = threading.Lock()
            _run_locks[run_id] = lock
        return lock


def request_pause(run_id: str) -> None:
    """Signal the worker for this run to stop scheduling new episodes."""
    _get_pause_event(run_id).set()


def request_resume(run_id: str) -> None:
    """Clear a pause flag so the worker can start scheduling episodes again."""
    _get_pause_event(run_id).clear()


def is_paused(run_id: str) -> bool:
    return _get_pause_event(run_id).is_set()


# ----------------------------------------------------------------------------
# Ingest: unpack uploaded zip into storage
# ----------------------------------------------------------------------------
def ingest_zip(storage: StorageBackend, run_id: str, zip_bytes: bytes) -> dict[str, list[str]]:
    """Save the zip and unpack the trajectories. Returns dict of categories ->
    list of episode names, each containing a trajectory.

    Writes are performed in parallel (settings.ingest_parallelism). On slow
    buckets (Rapid / HNS appendable) this is the difference between ~12 min
    and ~1-2 min for a zip of ~500 small objects.
    """
    run_pfx = manifest.run_prefix(run_id)
    storage.put_bytes(f"{run_pfx}/original.zip", zip_bytes, content_type="application/zip")

    discovered: dict[str, list[tuple[str, str]]] = {GOLDEN_CATEGORY: [], FAILURE_CATEGORY: []}
    writes: list[tuple[str, bytes]] = []  # (key, data)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if not parts:
                continue
            # find the Golden_/Failure_ segment anywhere in the path
            category = None
            seg_idx = None
            for i, seg in enumerate(parts):
                if seg in GOLDEN_DIRS:
                    category = GOLDEN_CATEGORY
                    seg_idx = i
                    break
                if seg in FAILURE_DIRS:
                    category = FAILURE_CATEGORY
                    seg_idx = i
                    break
            if category is None or seg_idx is None:
                continue
            if seg_idx + 2 >= len(parts):
                continue
            episode_name = parts[seg_idx + 1]
            sub_path = "/".join(parts[seg_idx + 1:])

            key = f"{run_pfx}/extracted/{parts[seg_idx]}/{sub_path}"
            with zf.open(info) as src:
                writes.append((key, src.read()))
            discovered[category].append((episode_name, key))

    # Write objects in parallel. Each one is independent (distinct key) so
    # they cannot contend; the bucket happily handles parallel streams.
    parallelism = max(1, get_settings().ingest_parallelism)
    log.info("ingest: writing %d objects with parallelism=%d", len(writes), parallelism)

    def _put(item: tuple[str, bytes]) -> None:
        k, data = item
        storage.put_bytes(k, data)

    if writes:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for _ in pool.map(_put, writes):
                pass

    # group by episode and select the trajectory file per episode
    out: dict[str, list[dict[str, str]]] = {GOLDEN_CATEGORY: [], FAILURE_CATEGORY: []}
    for category, items in discovered.items():
        per_ep: dict[str, list[str]] = {}
        for ep_name, key in items:
            per_ep.setdefault(ep_name, []).append(key)
        for ep_name in sorted(per_ep):
            traj_key = pick_trajectory_key(per_ep[ep_name])
            if traj_key:
                out[category].append({"name": ep_name, "trajectory_key": traj_key})
    return out  # type: ignore[return-value]


def pick_trajectory_key(keys: list[str]) -> str:
    """Selection rule from the plan:
      1. <episode>/agent/trajectory.json   (full Harbor layout)
      2. <episode>/trajectory.json         (bare)
      3. first *.json alphabetically       (fallback, with a warning)
    """
    by_basename: dict[str, str] = {Path(k).name: k for k in keys}
    # priority 1
    for k in keys:
        parts = Path(k).parts
        if len(parts) >= 2 and parts[-2] == "agent" and parts[-1] == "trajectory.json":
            return k
    # priority 2
    if "trajectory.json" in by_basename:
        return by_basename["trajectory.json"]
    # priority 3: first *.json alphabetically
    jsons = sorted(k for k in keys if k.endswith(".json"))
    if jsons:
        log.warning("falling back to %s as trajectory.json was not found", jsons[0])
        return jsons[0]
    return ""


# ----------------------------------------------------------------------------
# Helpers for per-episode metadata
# ----------------------------------------------------------------------------
def load_episode_meta(storage: StorageBackend, episode_dir_key: str) -> dict[str, Any]:
    """Look for result.json / config.json siblings to extract agent/model/task."""
    meta = {"task": "", "agent": "", "model": ""}
    for fname in ("result.json", "config.json"):
        key = f"{episode_dir_key}/{fname}"
        if not storage.exists(key):
            continue
        try:
            data = storage.get_json(key)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict):
            continue
        if fname == "result.json":
            meta["task"] = data.get("task_name") or meta["task"]
            ai = data.get("agent_info") or {}
            if ai.get("name"):
                meta["agent"] = ai["name"]
            mi = ai.get("model_info") or {}
            if mi.get("name"):
                meta["model"] = mi["name"]
        elif fname == "config.json":
            ag = data.get("agent") or {}
            meta["agent"] = meta["agent"] or ag.get("name", "")
            meta["model"] = meta["model"] or ag.get("model_name", "")
    return meta


def trajectory_dir_key(traj_key: str) -> str:
    """For …/episodeN/agent/trajectory.json -> …/episodeN. For
    …/episodeN/trajectory.json -> …/episodeN."""
    parts = traj_key.split("/")
    if len(parts) >= 2 and parts[-2] == "agent":
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


def load_trajectory(storage: StorageBackend, key: str) -> dict[str, Any]:
    raw = storage.get_text(key)
    return json.loads(raw)


# ----------------------------------------------------------------------------
# Worker entrypoint
# ----------------------------------------------------------------------------
def process_run(run_id: str) -> None:
    """Run the pipeline for one run. Idempotent at the episode level.

    Behaviour:
      - Per-run lock: only one process_run for the same run_id at a time.
      - Crash recovery: any episode left in 'running' from a prior worker is
        reset to 'pending' before scheduling.
      - Pause: if the in-memory pause event or the manifest's pause_requested
        flag is set, workers skip pending episodes and the run status flips
        to 'paused' (not 'done') at the end.
    """
    run_lock = _get_run_lock(run_id)
    if not run_lock.acquire(blocking=False):
        log.warning("process_run already in progress for %s; skipping", run_id)
        return

    try:
        storage = get_storage()
        settings = get_settings()

        try:
            m = manifest.read_manifest(storage, run_id)
        except Exception as e:  # noqa: BLE001
            log.exception("could not load manifest for %s: %s", run_id, e)
            return

        pause_event = _get_pause_event(run_id)
        # Reconcile in-process pause flag with the persisted flag (e.g. after
        # a uvicorn restart while paused).
        if m.get("pause_requested"):
            pause_event.set()
        else:
            pause_event.clear()

        # Crash recovery: any "running" episode left over from a prior worker
        # is stale (the run lock guarantees we're alone now). Reset to pending.
        for ep in m.get("episodes", []):
            if ep["status"] == "running":
                ep["status"] = "pending"
                if not ep.get("error_message"):
                    ep["error_message"] = "reset after worker restart"

        m["status"] = "running"
        if not m.get("started_at"):
            m["started_at"] = manifest.utc_now_iso()
        manifest.write_manifest(storage, m)
        manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))

        completed_normally = False
        try:
            client = evaluators.make_client(settings.openai_api_key)
            todo = [ep for ep in m["episodes"] if ep["status"] != "done"]
            parallelism = max(1, settings.max_parallel_episodes)
            log.info("processing %d episode(s) with parallelism=%d", len(todo), parallelism)

            manifest_lock = threading.Lock()

            def _run_one(ep: dict[str, Any]) -> None:
                # If pause was requested before we even started this episode,
                # don't begin — leave it pending so resume can pick it up.
                if pause_event.is_set():
                    return
                with manifest_lock:
                    ep["status"] = "running"
                    ep["started_at"] = manifest.utc_now_iso()
                    ep["error_message"] = ""
                    manifest.write_manifest(storage, m)

                try:
                    _process_episode(
                        storage, client, m, ep,
                        settings.openai_model, settings.openai_max_tokens,
                    )
                    ep["status"] = "done"
                except Exception as e:  # noqa: BLE001
                    log.exception("episode %s failed: %s", ep["episode_id"], e)
                    ep["status"] = "error"
                    ep["error_message"] = f"{type(e).__name__}: {e}"
                finally:
                    with manifest_lock:
                        ep["finished_at"] = manifest.utc_now_iso()
                        manifest.write_manifest(storage, m)
                        manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))

            if todo and not pause_event.is_set():
                # If pause is set partway, reflect that in the status promptly
                # so the UI shows "pausing" while in-flight episodes wrap up.
                pausing_announced = False
                with ThreadPoolExecutor(max_workers=parallelism) as pool:
                    futures = [pool.submit(_run_one, ep) for ep in todo]
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception:  # noqa: BLE001
                            log.exception("unexpected error from episode task")
                        if pause_event.is_set() and not pausing_announced:
                            with manifest_lock:
                                if m["status"] == "running":
                                    m["status"] = "pausing"
                                    manifest.write_manifest(storage, m)
                                    manifest.upsert_index_entry(
                                        storage, manifest.index_entry_from_manifest(m)
                                    )
                            pausing_announced = True

            # Decide terminal status for this invocation
            has_pending = any(ep["status"] == "pending" for ep in m["episodes"])
            if pause_event.is_set() and has_pending:
                # Paused with work remaining — do not write summaries yet.
                m["status"] = "paused"
            else:
                _write_summaries(storage, m)
                if not m["episodes"]:
                    m["status"] = "failed"
                    m["error_message"] = "no episodes discovered in zip"
                elif any(ep["status"] == "done" for ep in m["episodes"]):
                    m["status"] = "done"
                else:
                    m["status"] = "failed"
            completed_normally = True
        except Exception as e:  # noqa: BLE001
            log.exception("run %s failed: %s", run_id, e)
            m["status"] = "failed"
            m["error_message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        finally:
            # Only stamp finished_at if the run is truly terminal
            # (paused runs may still be resumed later).
            if m.get("status") in ("done", "failed"):
                m["finished_at"] = manifest.utc_now_iso()
            manifest.write_manifest(storage, m)
            manifest.upsert_index_entry(storage, manifest.index_entry_from_manifest(m))
            if completed_normally and m.get("status") != "paused":
                # Clear the in-process pause flag for terminal runs so we
                # don't keep stale state in memory forever.
                pause_event.clear()
    finally:
        run_lock.release()


def _process_episode(
    storage: StorageBackend,
    client: Any,
    m: dict[str, Any],
    ep: dict[str, Any],
    model: str,
    max_tokens: int,
) -> None:
    trajectory = load_trajectory(storage, ep["trajectory_key"])
    ep_dir = trajectory_dir_key(ep["trajectory_key"])
    meta = load_episode_meta(storage, ep_dir)

    # mirror inputs into a "results/..." path
    # turn "<run>/extracted/<cat_dir>/<rest>/trajectory.json" into
    # "<run>/results/<cat_dir>/<rest>/eval_*.json"
    parts = ep["trajectory_key"].split("/")
    try:
        idx = parts.index("extracted")
    except ValueError:
        idx = -1
    if idx >= 0:
        result_path = parts[: idx] + ["results"] + parts[idx + 1 : -1]
    else:
        result_path = parts[:-1]

    if ep["category"] == GOLDEN_CATEGORY:
        result = evaluators.evaluate_golden(client, trajectory, model, max_tokens)
        result_key = "/".join(result_path + ["eval_golden.json"])
        storage.put_json(result_key, result)
        ep["result_key"] = result_key
        ep["summary"] = evaluators.golden_summary_row(
            result,
            task=meta["task"] or m["task_name"],
            agent=meta["agent"],
            model=meta["model"] or m["model"],
            trajectory_name=ep["trajectory_key"],
        )
    else:
        # for failure episodes, also pull verifier/test-stdout.txt + exception.txt if present
        failure_output = _maybe_read(storage, f"{ep_dir}/verifier/test-stdout.txt", limit=1000)
        exception_text = _maybe_read(storage, f"{ep_dir}/exception.txt", limit=8000)
        eval_out = evaluators.evaluate_failure(
            client,
            trajectory,
            model,
            agent=meta["agent"] or "unknown",
            task=meta["task"] or m["task_name"],
            failure_output=failure_output,
            exception=exception_text,
        )
        # Also classify the trajectory shape with the GT rubric — only the class
        # label + 1-sentence justification are surfaced. The rest of the GT
        # output (SCs, hard requirements, verdict) is discarded for failures.
        gt_class = ""
        gt_justification = ""
        try:
            gt_result = evaluators.evaluate_golden(client, trajectory, model, max_tokens)
            gt_block = gt_result.get("gt_class") or {}
            gt_class = gt_block.get("label") or ""
            gt_justification = gt_block.get("justification") or ""
        except Exception as e:  # noqa: BLE001
            log.warning("GT classification failed for failure episode %s: %s", ep["episode_id"], e)

        result_key = "/".join(result_path + ["eval_failure.json"])
        storage.put_json(result_key, eval_out)
        ep["result_key"] = result_key
        ep["summary"] = evaluators.failure_summary_row(
            eval_out,
            agent=meta["agent"] or "unknown",
            task=meta["task"] or m["task_name"],
            episode_name=ep["name"],
            status="fail",
            gt_class=gt_class,
            gt_justification=gt_justification,
        )


def _maybe_read(storage: StorageBackend, key: str, limit: int = 1000) -> str:
    if not storage.exists(key):
        return ""
    try:
        return storage.get_text(key)[:limit]
    except Exception:  # noqa: BLE001
        return ""


# ----------------------------------------------------------------------------
# Per-run summary spreadsheets (mirror what the CLI writes)
# ----------------------------------------------------------------------------
GOLDEN_COLUMNS = [
    "Task", "Agent", "Model",
    "GT Class(AI)", "GT Justification(AI)", "Success Criteria (AI)",
    "GT Class(Human)", "Success Criteria (Human)",
    "HITL Remarks", "Task name", "Trajectory name",
]

FAILURE_COLUMNS = [
    "agent", "benchmark", "trial_id", "status",
    "GT Class(AI)", "GT Justification(AI)",
    "failure_type", "reason", "root_cause", "fix",
]


def _write_summaries(storage: StorageBackend, m: dict[str, Any]) -> None:
    run_pfx = manifest.run_prefix(m["run_id"])

    golden_rows = [ep["summary"] for ep in m["episodes"]
                   if ep["category"] == GOLDEN_CATEGORY and ep["status"] == "done"]
    failure_rows = [ep["summary"] for ep in m["episodes"]
                    if ep["category"] == FAILURE_CATEGORY and ep["status"] == "done"]

    # Golden -> CSV
    if golden_rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=GOLDEN_COLUMNS)
        writer.writeheader()
        for row in golden_rows:
            writer.writerow({k: row.get(k, "") for k in GOLDEN_COLUMNS})
        storage.put_text(f"{run_pfx}/golden_summary.csv", buf.getvalue(), content_type="text/csv")

    # Failure -> XLSX
    if failure_rows:
        df = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        storage.put_bytes(
            f"{run_pfx}/failure_summary.xlsx",
            buf.getvalue(),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    m["total_episodes"] = len(m["episodes"])
    m["golden_count"] = sum(1 for ep in m["episodes"] if ep["category"] == GOLDEN_CATEGORY)
    m["failure_count"] = sum(1 for ep in m["episodes"] if ep["category"] == FAILURE_CATEGORY)
