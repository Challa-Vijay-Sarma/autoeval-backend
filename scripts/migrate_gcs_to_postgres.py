"""One-shot backfill: existing GCS-stored runs -> Postgres rows.

Reads the legacy index.json + per-run manifest.json from the configured
storage backend and upserts the equivalent rows into Postgres. Safe to re-run:
- An existing run with finished_at >= manifest's finished_at is skipped.
- Per-run work happens in a single transaction.

Usage:
    cd backend
    python -m scripts.migrate_gcs_to_postgres [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from platform_app.config import get_settings
from platform_app.db import sync_session
from platform_app.models import Episode, Run
from platform_app.storage import StorageBackend, get_storage, storage_uri_for


log = logging.getLogger("migrate")


INDEX_KEY = "index.json"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v)
    except Exception:  # noqa: BLE001
        return None


def _coerce_run_id(raw: str) -> uuid.UUID:
    """Old run_id was uuid4().hex (32 chars, no hyphens). Accept both."""
    return uuid.UUID(raw)


def _key_if_exists(storage: StorageBackend, key: str) -> str | None:
    return storage_uri_for(storage, key) if storage.exists(key) else None


def _read_index(storage: StorageBackend) -> list[dict[str, Any]]:
    if not storage.exists(INDEX_KEY):
        return []
    data = storage.get_json(INDEX_KEY)
    return data if isinstance(data, list) else []


def _read_manifest(storage: StorageBackend, run_id_hex: str) -> dict[str, Any] | None:
    key = f"runs/{run_id_hex}/manifest.json"
    if not storage.exists(key):
        return None
    obj = storage.get_json(key)
    return obj if isinstance(obj, dict) else None


def _upsert_run(
    session,
    storage: StorageBackend,
    manifest: dict[str, Any],
    *,
    dry_run: bool,
) -> bool:
    raw_id = manifest.get("run_id")
    if not raw_id:
        log.warning("manifest missing run_id; skipping")
        return False
    try:
        rid = _coerce_run_id(raw_id)
    except ValueError:
        log.warning("manifest run_id %r is not a UUID; skipping", raw_id)
        return False

    existing = session.execute(select(Run).where(Run.id == rid)).scalar_one_or_none()

    manifest_finished = _parse_iso(manifest.get("finished_at"))
    if existing is not None and manifest_finished is not None and existing.finished_at is not None:
        if existing.finished_at >= manifest_finished:
            log.info("skip %s (already up-to-date)", raw_id)
            return False

    run_pfx = f"runs/{rid.hex}"
    new_values = dict(
        task_name=manifest.get("task_name") or "",
        uploaded_filename=manifest.get("uploaded_filename") or "",
        model=manifest.get("model") or "",
        status=manifest.get("status") or "queued",
        pause_requested=bool(manifest.get("pause_requested", False)),
        total_episodes=int(manifest.get("total_episodes") or 0),
        golden_count=int(manifest.get("golden_count") or 0),
        failure_count=int(manifest.get("failure_count") or 0),
        error_message=manifest.get("error_message") or "",
        created_at=_parse_iso(manifest.get("created_at")) or datetime.now(timezone.utc),
        started_at=_parse_iso(manifest.get("started_at")),
        finished_at=manifest_finished,
        original_zip_uri=_key_if_exists(storage, f"{run_pfx}/original.zip"),
        summary_csv_uri=_key_if_exists(storage, f"{run_pfx}/golden_summary.csv"),
        summary_xlsx_uri=_key_if_exists(storage, f"{run_pfx}/failure_summary.xlsx"),
        explorer_html_uri=_key_if_exists(storage, f"{run_pfx}/explorer.html"),
    )

    if dry_run:
        action = "update" if existing else "insert"
        log.info("[dry-run] would %s run %s", action, raw_id)
        return True

    if existing is None:
        run = Run(id=rid, **new_values)
        session.add(run)
    else:
        for k, v in new_values.items():
            setattr(existing, k, v)
        run = existing
    session.flush()

    # Reset episodes for this run (idempotent rebuild).
    session.query(Episode).filter(Episode.run_id == rid).delete()
    session.flush()

    for ep in manifest.get("episodes") or []:
        if not isinstance(ep, dict):
            continue
        category = ep.get("category") or ""
        name = ep.get("name") or ""
        traj_key = ep.get("trajectory_key") or ""
        if not (category and name and traj_key):
            continue

        result_key = ep.get("result_key") or ""
        result_obj: dict[str, Any] | None = None
        if result_key and storage.exists(result_key):
            try:
                fetched = storage.get_json(result_key)
                if isinstance(fetched, dict):
                    result_obj = fetched
            except Exception:  # noqa: BLE001
                log.warning("could not load result %s for ep %s", result_key, name)

        episode = Episode(
            run_id=rid,
            episode_key=ep.get("episode_id") or f"{category}__{name}",
            category=category,
            name=name,
            trajectory_uri=storage_uri_for(storage, traj_key),
            result_uri=storage_uri_for(storage, result_key) if result_key else None,
            status=ep.get("status") or "pending",
            error_message=ep.get("error_message") or "",
            result=result_obj,
            summary=ep.get("summary") or {},
            started_at=_parse_iso(ep.get("started_at")),
            finished_at=_parse_iso(ep.get("finished_at")),
        )
        session.add(episode)
    session.flush()
    log.info("imported run %s (%d episodes)", raw_id, len(manifest.get("episodes") or []))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="cap on runs to import (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    storage = get_storage(settings)

    index_entries = _read_index(storage)
    if not index_entries:
        log.info("no runs found in index.json — nothing to do")
        return 0

    log.info("found %d runs in index.json", len(index_entries))
    if args.limit:
        index_entries = index_entries[: args.limit]

    imported = 0
    for entry in index_entries:
        raw_id = entry.get("run_id")
        if not raw_id:
            continue
        manifest = _read_manifest(storage, raw_id)
        if manifest is None:
            log.warning("manifest missing for run %s; skipping", raw_id)
            continue

        with sync_session() as session:
            if _upsert_run(session, storage, manifest, dry_run=args.dry_run):
                imported += 1

    log.info("done: %d run(s) %s", imported, "would import" if args.dry_run else "imported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
