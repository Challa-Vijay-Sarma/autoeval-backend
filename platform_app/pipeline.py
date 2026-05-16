"""Zip ingestion + episode discovery + background worker (Postgres-backed).

All run/episode state lives in Postgres. The worker uses `claim_pending` with
`FOR UPDATE SKIP LOCKED` so multiple Cloud Run instances can process the same
run safely. The only in-process state remaining is the OpenAI client object
and the thread pool; no run-keyed locks or pause events.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import evaluators
from .config import get_settings
from .db import sync_session
from .models import Run
from .repositories import EpisodesRepository, NewEpisode, RunsRepository
from .storage import StorageBackend, get_storage, storage_uri_for


log = logging.getLogger("autoeval.pipeline")


GOLDEN_CATEGORY = "golden"
FAILURE_CATEGORY = "failure"
GOLDEN_DIRS = {"Golden_trajectories", "golden_trajectories"}
FAILURE_DIRS = {"Failure_trajectories", "failure_trajectories"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_prefix(run_id: str | uuid.UUID) -> str:
    rid = run_id.hex if isinstance(run_id, uuid.UUID) else run_id
    return f"runs/{rid}"


# ----------------------------------------------------------------------------
# Ingest: unpack uploaded zip into storage
# ----------------------------------------------------------------------------
def ingest_zip(
    storage: StorageBackend, run_id: str | uuid.UUID, zip_bytes: bytes
) -> dict[str, list[dict[str, str]]]:
    """Save the zip and unpack the trajectories.

    Returns {category: [{name, trajectory_key}]}. The caller persists these
    via EpisodesRepository.bulk_insert.
    """
    run_pfx = run_prefix(run_id)
    storage.put_bytes(f"{run_pfx}/original.zip", zip_bytes, content_type="application/zip")

    discovered: dict[str, list[tuple[str, str]]] = {GOLDEN_CATEGORY: [], FAILURE_CATEGORY: []}
    writes: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if not parts:
                continue
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

    parallelism = max(1, get_settings().ingest_parallelism)
    log.info("ingest: writing %d objects with parallelism=%d", len(writes), parallelism)

    def _put(item: tuple[str, bytes]) -> None:
        k, data = item
        storage.put_bytes(k, data)

    if writes:
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            for _ in pool.map(_put, writes):
                pass

    out: dict[str, list[dict[str, str]]] = {GOLDEN_CATEGORY: [], FAILURE_CATEGORY: []}
    for category, items in discovered.items():
        per_ep: dict[str, list[str]] = {}
        for ep_name, key in items:
            per_ep.setdefault(ep_name, []).append(key)
        for ep_name in sorted(per_ep):
            traj_key = pick_trajectory_key(per_ep[ep_name])
            if traj_key:
                out[category].append({"name": ep_name, "trajectory_key": traj_key})
    return out


def pick_trajectory_key(keys: list[str]) -> str:
    """1) <ep>/agent/trajectory.json  2) <ep>/trajectory.json  3) first *.json."""
    by_basename: dict[str, str] = {Path(k).name: k for k in keys}
    for k in keys:
        parts = Path(k).parts
        if len(parts) >= 2 and parts[-2] == "agent" and parts[-1] == "trajectory.json":
            return k
    if "trajectory.json" in by_basename:
        return by_basename["trajectory.json"]
    jsons = sorted(k for k in keys if k.endswith(".json"))
    if jsons:
        log.warning("falling back to %s as trajectory.json was not found", jsons[0])
        return jsons[0]
    return ""


# ----------------------------------------------------------------------------
# Per-episode metadata helpers
# ----------------------------------------------------------------------------
def load_episode_meta(storage: StorageBackend, episode_dir_key: str) -> dict[str, Any]:
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
    parts = traj_key.split("/")
    if len(parts) >= 2 and parts[-2] == "agent":
        return "/".join(parts[:-2])
    return "/".join(parts[:-1])


def load_trajectory(storage: StorageBackend, key: str) -> dict[str, Any]:
    return json.loads(storage.get_text(key))


# ----------------------------------------------------------------------------
# Episode processing result
# ----------------------------------------------------------------------------
@dataclass
class EpisodeOutcome:
    result_uri: str
    result_key: str
    result: dict[str, Any]
    summary: dict[str, Any]


# ----------------------------------------------------------------------------
# Worker entrypoint
# ----------------------------------------------------------------------------
def process_run(run_id: str | uuid.UUID) -> None:
    """Run the pipeline for one run.

    Idempotent and safe across multiple instances:
      - `claim_pending(... FOR UPDATE SKIP LOCKED)` guarantees no episode is
        processed twice even with concurrent workers.
      - `pause_requested` re-read between claims is the pause source of truth.
      - Stale 'running' episodes (>15 min) are reset to 'pending' at startup.
    """
    rid = run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(run_id)
    settings = get_settings()
    storage = get_storage()

    # Load run + reset stale running episodes + flip to 'running'
    with sync_session() as session:
        runs_repo = RunsRepository(session)
        eps_repo = EpisodesRepository(session)
        run = runs_repo.get(rid, with_episodes=False)
        if run is None:
            log.warning("process_run: run %s not found", rid)
            return
        if run.pause_requested or run.status in ("paused", "pausing"):
            log.info("process_run: run %s is paused; not starting worker", rid)
            return

        eps_repo.reset_stale_running(
            rid, stale_after_seconds=settings.episode_stale_after_seconds
        )
        runs_repo.update_status(
            rid,
            "running",
            started_at=run.started_at or _utc_now(),
        )

    parallelism = max(1, settings.max_parallel_episodes)
    client = evaluators.make_client(settings.openai_api_key)

    # Snapshot of per-run constants needed inside worker threads.
    with sync_session() as session:
        run = RunsRepository(session).get(rid, with_episodes=False)
        if run is None:
            return
        run_task_name = run.task_name
        run_model = run.model

    completed_normally = False
    try:
        # Each worker thread independently claims episodes until none remain
        # or the run is paused.
        def _worker_loop() -> None:
            while True:
                # Pause check
                with sync_session() as s:
                    r = RunsRepository(s).get(rid, with_episodes=False)
                    if r is None or r.pause_requested:
                        return

                # Claim one
                with sync_session() as s:
                    eps = EpisodesRepository(s).claim_pending(rid, limit=1)
                if not eps:
                    return
                ep = eps[0]
                ep_id = ep.id

                try:
                    outcome = _process_episode(
                        storage,
                        client,
                        run_task_name=run_task_name,
                        run_model=run_model,
                        run_id=rid,
                        category=ep.category,
                        ep_name=ep.name,
                        trajectory_key=_uri_to_key(ep.trajectory_uri, storage),
                        model=settings.openai_model,
                        max_tokens=settings.openai_max_tokens,
                    )
                except Exception as e:  # noqa: BLE001
                    log.exception("episode %s failed: %s", ep.episode_key, e)
                    with sync_session() as s:
                        EpisodesRepository(s).mark_error(
                            ep_id, f"{type(e).__name__}: {e}"
                        )
                    continue

                with sync_session() as s:
                    EpisodesRepository(s).mark_done(
                        ep_id,
                        result=outcome.result,
                        summary=outcome.summary,
                        result_uri=outcome.result_uri,
                    )

        # Spawn the pool. Each thread loops until exhausted.
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = [pool.submit(_worker_loop) for _ in range(parallelism)]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:  # noqa: BLE001
                    log.exception("worker thread crashed")

        # Decide terminal status
        with sync_session() as session:
            runs_repo = RunsRepository(session)
            eps_repo = EpisodesRepository(session)
            run = runs_repo.get(rid, with_episodes=True)
            if run is None:
                return
            episodes = list(run.episodes)
            has_pending = any(ep.status == "pending" for ep in episodes)
            paused = run.pause_requested

            if paused and has_pending:
                runs_repo.update_status(rid, "paused")
            else:
                # Write summaries + recompute counts before flipping to terminal.
                _write_summaries(storage, run, episodes)
                _record_summary_uris(runs_repo, storage, rid, episodes)
                runs_repo.recompute_counts(rid)

                if not episodes:
                    runs_repo.update_status(
                        rid,
                        "failed",
                        error_message="no episodes discovered in zip",
                        finished_at=_utc_now(),
                    )
                elif any(ep.status == "done" for ep in episodes):
                    runs_repo.update_status(rid, "done", finished_at=_utc_now())
                else:
                    runs_repo.update_status(rid, "failed", finished_at=_utc_now())

        # Build per-episode explorer.html files in a daemon thread (slow on remote
        # buckets; the run is already marked terminal so the UI doesn't wait).
        with sync_session() as session:
            run = RunsRepository(session).get(rid, with_episodes=False)
        if run is not None and run.status == "done":
            threading.Thread(
                target=_write_run_episode_explorers,
                args=(rid,),
                daemon=True,
                name=f"ep-explorers-{rid.hex[:8]}",
            ).start()

        completed_normally = True
    except Exception as e:  # noqa: BLE001
        log.exception("run %s failed: %s", rid, e)
        with sync_session() as session:
            RunsRepository(session).update_status(
                rid,
                "failed",
                error_message=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                finished_at=_utc_now(),
            )
    _ = completed_normally  # kept for future telemetry hooks


def _uri_to_key(uri: str, storage: StorageBackend) -> str:
    """Convert a stored URI back to a key the storage backend understands."""
    from .storage import key_from_uri
    return key_from_uri(uri)


# ----------------------------------------------------------------------------
# Per-episode processing
# ----------------------------------------------------------------------------
def _process_episode(
    storage: StorageBackend,
    client: Any,
    *,
    run_task_name: str,
    run_model: str,
    run_id: uuid.UUID,
    category: str,
    ep_name: str,
    trajectory_key: str,
    model: str,
    max_tokens: int,
) -> EpisodeOutcome:
    trajectory = load_trajectory(storage, trajectory_key)
    ep_dir = trajectory_dir_key(trajectory_key)
    meta = load_episode_meta(storage, ep_dir)

    # mirror inputs into a "results/..." path
    parts = trajectory_key.split("/")
    try:
        idx = parts.index("extracted")
    except ValueError:
        idx = -1
    if idx >= 0:
        result_path_parts = parts[:idx] + ["results"] + parts[idx + 1 : -1]
    else:
        result_path_parts = parts[:-1]

    # Predict where this episode's HTML will land inside the results.zip bundle.
    # The bundle layout is `{category}/{safe(name)}.html`; the path is
    # deterministic from (category, name) so we can write it into the summary
    # row even before the explorer daemon thread has produced the HTML.
    from .naming import safe_filename as _safe_filename
    explorer_path = f"{category}/{_safe_filename(ep_name)}.html"

    if category == GOLDEN_CATEGORY:
        result = evaluators.evaluate_golden(client, trajectory, model, max_tokens)
        result_key = "/".join(result_path_parts + ["eval_golden.json"])
        storage.put_json(result_key, result)
        summary = evaluators.golden_summary_row(
            result,
            task=meta["task"] or run_task_name,
            agent=meta["agent"],
            model=meta["model"] or run_model,
            trajectory_name=trajectory_key,
            explorer_path=explorer_path,
        )
    else:
        failure_output = _maybe_read(storage, f"{ep_dir}/verifier/test-stdout.txt", limit=1000)
        exception_text = _maybe_read(storage, f"{ep_dir}/exception.txt", limit=8000)
        eval_out = evaluators.evaluate_failure(
            client,
            trajectory,
            model,
            agent=meta["agent"] or "unknown",
            task=meta["task"] or run_task_name,
            failure_output=failure_output,
            exception=exception_text,
        )
        gt_class = ""
        gt_justification = ""
        try:
            gt_result = evaluators.evaluate_golden(client, trajectory, model, max_tokens)
            gt_block = gt_result.get("gt_class") or {}
            gt_class = gt_block.get("label") or ""
            gt_justification = gt_block.get("justification") or ""
        except Exception as e:  # noqa: BLE001
            log.warning("GT classification failed for failure episode %s: %s", ep_name, e)

        result_key = "/".join(result_path_parts + ["eval_failure.json"])
        storage.put_json(result_key, eval_out)
        result = eval_out
        summary = evaluators.failure_summary_row(
            eval_out,
            agent=meta["agent"] or "unknown",
            task=meta["task"] or run_task_name,
            episode_name=ep_name,
            status="fail",
            gt_class=gt_class,
            gt_justification=gt_justification,
            explorer_path=explorer_path,
        )

    return EpisodeOutcome(
        result_uri=storage_uri_for(storage, result_key),
        result_key=result_key,
        result=result,
        summary=summary,
    )


def _maybe_read(storage: StorageBackend, key: str, limit: int = 1000) -> str:
    if not storage.exists(key):
        return ""
    try:
        return storage.get_text(key)[:limit]
    except Exception:  # noqa: BLE001
        return ""


# ----------------------------------------------------------------------------
# Per-run summary spreadsheets
# ----------------------------------------------------------------------------
GOLDEN_COLUMNS = [
    "Task", "Agent", "Model",
    "GT Class(AI)", "GT Justification(AI)", "Success Criteria (AI)",
    "GT Class(Human)", "Success Criteria (Human)",
    "HITL Remarks", "Task name", "Trajectory name",
    "Explorer HTML",
]

FAILURE_COLUMNS = [
    "agent", "benchmark", "trial_id", "status",
    "GT Class(AI)", "GT Justification(AI)",
    "failure_type", "reason", "root_cause", "fix",
    "Explorer HTML",
]


def _write_summaries(storage: StorageBackend, run: Run, episodes: list) -> None:
    run_pfx = run_prefix(run.id)

    golden_rows = [
        ep.summary for ep in episodes
        if ep.category == GOLDEN_CATEGORY and ep.status == "done"
    ]
    failure_rows = [
        ep.summary for ep in episodes
        if ep.category == FAILURE_CATEGORY and ep.status == "done"
    ]

    if golden_rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=GOLDEN_COLUMNS)
        writer.writeheader()
        for row in golden_rows:
            writer.writerow({k: row.get(k, "") for k in GOLDEN_COLUMNS})
        storage.put_text(
            f"{run_pfx}/golden_summary.csv", buf.getvalue(), content_type="text/csv"
        )

    if failure_rows:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow({k: row.get(k, "") for k in FAILURE_COLUMNS})
        storage.put_text(
            f"{run_pfx}/failure_summary.csv", buf.getvalue(), content_type="text/csv"
        )


def _record_summary_uris(
    runs_repo: RunsRepository,
    storage: StorageBackend,
    rid: uuid.UUID,
    episodes: list,
) -> None:
    run_pfx = run_prefix(rid)
    golden_key = f"{run_pfx}/golden_summary.csv"
    # NB: the DB column `summary_xlsx_uri` is a legacy name; now holds the path
    # to the failure CSV (the file is no longer an .xlsx).
    failure_key = f"{run_pfx}/failure_summary.csv"
    zip_key = f"{run_pfx}/original.zip"
    csv_uri = storage_uri_for(storage, golden_key) if storage.exists(golden_key) else None
    xlsx_uri = storage_uri_for(storage, failure_key) if storage.exists(failure_key) else None
    zip_uri = storage_uri_for(storage, zip_key) if storage.exists(zip_key) else None
    runs_repo.set_artifacts(
        rid,
        original_zip_uri=zip_uri,
        summary_csv_uri=csv_uri,
        summary_xlsx_uri=xlsx_uri,
    )


# ----------------------------------------------------------------------------
# Per-episode explorer.html generation
# ----------------------------------------------------------------------------
_EPISODE_EXPLORER_PARALLELISM = 4

# The (run-level) nav block emitted by `templates/explorer_template.html`. We
# replace this block wholesale with our own data-driven nav. Kept identical to
# the template — any drift here will surface as a RuntimeError at generation.
_EXPLORER_OLD_NAV = (
    '  const nav = document.getElementById("nav");\n'
    '  const lbl = document.createElement("div");\n'
    '  lbl.className = "section-label";\n'
    '  lbl.textContent = "Agent run";\n'
    '  nav.appendChild(lbl);\n'
    '  nav.appendChild(makeFolder("agent", ["agent/trajectory.json", "agent/test-stdout.txt"]));\n'
    '\n'
    '  const lbl2 = document.createElement("div");\n'
    '  lbl2.className = "section-label";\n'
    '  lbl2.textContent = "Task tree";\n'
    '  nav.appendChild(lbl2);\n'
    '  nav.appendChild(makeFileLink("instruction.md", "instruction.md"));\n'
    '  nav.appendChild(makeFolder("environment", Object.keys(FILES).filter((k) => k.startsWith("environment/"))));\n'
    '  nav.appendChild(makeFolder("solution", Object.keys(FILES).filter((k) => k.startsWith("solution/"))));\n'
    '  nav.appendChild(makeFolder("tests", Object.keys(FILES).filter((k) => k.startsWith("tests/"))));'
)


def _write_run_episode_explorers(rid: uuid.UUID) -> None:
    """Daemon-thread entrypoint. Generates a focused HTML per episode."""
    storage = get_storage()

    # Pull a snapshot of episodes; we close the session before doing I/O.
    with sync_session() as session:
        run = RunsRepository(session).get(rid, with_episodes=True)
        if run is None:
            return
        items = [
            (ep.id, ep.episode_key, ep.category, ep.name, ep.trajectory_uri)
            for ep in run.episodes
        ]
        run_title = run.task_name or rid.hex

    # Resolve template + bfe once per run.
    settings = get_settings()
    project_root_path = settings_project_root_path()
    template_path = project_root_path / settings.explorer_template_path
    if not template_path.is_file():
        log.warning("explorer template not found at %s; skipping episode explorers", template_path)
        return
    template_html = template_path.read_text(encoding="utf-8")

    def _one(item: tuple) -> None:
        ep_id, ep_key, category, name, traj_uri = item
        try:
            uri = _write_episode_explorer(
                storage,
                run_id=rid,
                episode_id=ep_id,
                category=category,
                name=name,
                trajectory_uri=traj_uri,
                run_title=run_title,
                template_html=template_html,
            )
        except Exception:  # noqa: BLE001
            log.exception("episode-explorer failed for %s", ep_key)
            return
        if uri:
            with sync_session() as session:
                EpisodesRepository(session).set_explorer_uri(ep_id, uri)

    if not items:
        return
    with ThreadPoolExecutor(max_workers=_EPISODE_EXPLORER_PARALLELISM) as pool:
        for _ in pool.map(_one, items):
            pass


def settings_project_root_path() -> Path:
    from . import config as _config
    return _config.project_root()


def _write_episode_explorer(
    storage: StorageBackend,
    *,
    run_id: uuid.UUID,
    episode_id: uuid.UUID,
    category: str,
    name: str,
    trajectory_uri: str,
    run_title: str,
    template_html: str,
) -> str | None:
    """Build a focused, self-contained HTML for one episode.

    Embeds trajectory.json plus, when present, verifier/test-stdout.txt and
    exception.txt — and nothing else.
    """
    from .storage import key_from_uri

    traj_key = key_from_uri(trajectory_uri)
    if not traj_key:
        return None
    try:
        traj_text = storage.get_text(traj_key)
    except Exception:  # noqa: BLE001
        log.exception("could not read trajectory %s", traj_key)
        return None

    ep_dir = trajectory_dir_key(traj_key)

    file_map: dict[str, str] = {"trajectory.json": traj_text}

    stdout_key = f"{ep_dir}/verifier/test-stdout.txt"
    if storage.exists(stdout_key):
        try:
            file_map["verifier/test-stdout.txt"] = storage.get_text(stdout_key)
        except Exception:  # noqa: BLE001
            log.warning("test-stdout.txt unreadable for %s", stdout_key)

    exc_key = f"{ep_dir}/exception.txt"
    if storage.exists(exc_key):
        try:
            file_map["exception.txt"] = storage.get_text(exc_key)
        except Exception:  # noqa: BLE001
            log.warning("exception.txt unreadable for %s", exc_key)

    nav_data = {
        "sections": [
            {"label": name, "files": list(file_map.keys())},
        ]
    }

    title = f"{run_title} · {name}"
    subtitle = f"{category} episode"
    pill = category

    html_text = _patch_explorer_template_for_run(
        template_html,
        title=title,
        subtitle=subtitle,
        pill=pill,
        file_map=file_map,
        nav_data=nav_data,
    )

    out_key = f"runs/{run_id.hex}/episodes/{episode_id.hex}/explorer.html"
    storage.put_bytes(
        out_key,
        html_text.encode("utf-8"),
        content_type="text/html; charset=utf-8",
    )
    log.info(
        "wrote %s (%d files, %d bytes)", out_key, len(file_map), len(html_text)
    )
    return storage_uri_for(storage, out_key)


def _patch_explorer_template_for_run(
    html: str,
    *,
    title: str,
    subtitle: str,
    pill: str,
    file_map: dict[str, str],
    nav_data: dict[str, Any],
) -> str:
    import re as _re
    import sys as _sys

    from . import config as _config
    project_root_str = str(_config.project_root())
    if project_root_str not in _sys.path:
        _sys.path.insert(0, project_root_str)
    import build_folder_explorer as bfe  # type: ignore[import-not-found]

    embed = bfe.json_for_script_tag(file_map)
    start_tag = '<script type="application/json" id="file-embed">'
    i = html.find(start_tag)
    if i == -1:
        raise RuntimeError("Template missing file-embed script tag.")
    j = html.find("</script>", i)
    if j == -1:
        raise RuntimeError("Template malformed: no closing </script> for file-embed.")
    j += len("</script>")
    html = html[:i] + start_tag + embed + "</script>" + html[j:]

    html = _re.sub(
        r"<title>.*?</title>",
        f"<title>{_escape_xml(title)}</title>",
        html,
        count=1,
        flags=_re.DOTALL,
    )

    html = _re.sub(
        r'(<div class="brand">\s*<h1>)(.*?)(</h1>\s*<p>)(.*?)(</p>\s*<span class="pill">)(.*?)(</span>\s*</div>)',
        rf"\g<1>{_escape_xml(title)}\g<3>{_escape_xml(subtitle)}\g<5>{_escape_xml(pill)}\g<7>",
        html,
        count=1,
        flags=_re.DOTALL,
    )

    nav_json = bfe.json_for_script_tag(nav_data)
    new_nav = (
        '  const nav = document.getElementById("nav");\n'
        f'  const navData = {nav_json};\n'
        '  navData.sections.forEach((section) => {\n'
        '    if (!section) return;\n'
        '    const lbl = document.createElement("div");\n'
        '    lbl.className = "section-label";\n'
        '    lbl.textContent = section.label;\n'
        '    nav.appendChild(lbl);\n'
        '    (section.files || []).forEach((p) => {\n'
        '      const label = p.includes("/") ? p.split("/").pop() : p;\n'
        '      nav.appendChild(makeFileLink(p, label));\n'
        '    });\n'
        '    (section.folders || []).forEach((f) => {\n'
        '      const folder = makeFolder(f.title, f.children || []);\n'
        '      if (f.open === false) {\n'
        '        const head = folder.querySelector(".folder-btn");\n'
        '        const kids = folder.querySelector(".children");\n'
        '        if (head) head.classList.remove("open");\n'
        '        if (kids) kids.classList.remove("open");\n'
        '      }\n'
        '      nav.appendChild(folder);\n'
        '    });\n'
        '  });'
    )
    if _EXPLORER_OLD_NAV not in html:
        raise RuntimeError(
            "Template no longer matches expected nav block; refresh "
            "templates/explorer_template.html or update _EXPLORER_OLD_NAV."
        )
    return html.replace(_EXPLORER_OLD_NAV, new_nav, 1)


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
