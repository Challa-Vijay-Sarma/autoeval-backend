"""Storage abstraction: local filesystem (dev) or GCS bucket (prod).

The interface mirrors the small subset of operations the platform needs:
  - put_bytes / put_text / put_json
  - get_bytes / get_text / get_json
  - exists, list_prefix, delete_prefix
  - open_writable_stream (for streamed uploads of the zip)

Paths are always slash-separated and bucket-relative. Local backend maps
"foo/bar" -> <storage_local_dir>/foo/bar.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO, Iterable, Protocol

from .config import Settings, get_settings

log = logging.getLogger("autoeval.storage")

# Rapid/zonal buckets require gRPC appendable-object writes; asyncio.run() cannot be used
# from inside FastAPI's event loop, so we run the coroutine in a dedicated thread.
_APPEND_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="gcs-appendable"
)
# get_storage() returns a new GCSStorage per call; persist appendable mode per bucket globally.
_APPENDABLE_BUCKETS: set[str] = set()


def _exception_chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and len(out) < 12:
        out.append(e)
        e = e.__cause__ or getattr(e, "__context__", None)
    return out


def _error_text_requires_appendable_bucket(exc: BaseException) -> bool:
    parts: list[str] = []
    for e in _exception_chain(exc):
        msg = getattr(e, "message", None)
        if msg:
            parts.append(str(msg))
        parts.append(str(e))
    blob = " ".join(parts).lower()
    return "appendable" in blob and "bucket" in blob


def _appendable_put_sync(bucket_name: str, object_name: str, data: bytes) -> None:
    """Write object via Storage gRPC appendable-object API (required for Rapid/zonal buckets)."""

    from google.cloud.storage.asyncio.async_appendable_object_writer import (
        AsyncAppendableObjectWriter,
    )
    from google.cloud.storage.asyncio.async_grpc_client import AsyncGrpcClient

    async def _run() -> None:
        grpc_cli = AsyncGrpcClient()
        try:
            writer = AsyncAppendableObjectWriter(
                grpc_cli,
                bucket_name,
                object_name,
                generation=None,
            )
            await writer.open()
            if data:
                await writer.append(data)
            await writer.flush()
            await writer.close(finalize_on_close=True)
        finally:
            # Drain the gRPC client's background stream tasks before the event
            # loop closes. Without this, asyncio.run() tears the loop down with
            # pending `_consume_request_iterator` tasks and logs noisy warnings
            # ("Task was destroyed but it is pending!"). Different versions of
            # google-cloud-storage expose this method under different names; try
            # the common ones in order and swallow failures (best-effort cleanup).
            for fn_name in ("close", "aclose"):
                fn = getattr(grpc_cli, fn_name, None)
                if not callable(fn):
                    continue
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                    break
                except Exception:  # noqa: BLE001
                    log.debug("AsyncGrpcClient.%s() failed; ignored", fn_name, exc_info=True)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        fut = _APPEND_EXECUTOR.submit(lambda: asyncio.run(_run()))
        fut.result(timeout=600)


class StorageBackend(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> None: ...
    def put_json(self, key: str, obj: object) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def get_text(self, key: str) -> str: ...
    def get_json(self, key: str) -> object: ...
    def exists(self, key: str) -> bool: ...
    def list_prefix(self, prefix: str) -> Iterable[str]: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def stream_writable(self, key: str, content_type: str = "application/octet-stream") -> IO[bytes]: ...
    def stream_readable(self, key: str) -> IO[bytes]: ...
    def local_path(self, key: str) -> Path | None: ...
    """Local-only escape hatch; returns None on remote backends."""


# ----------------------------------------------------------------------------
# Local FS implementation
# ----------------------------------------------------------------------------
class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp + rename
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type)

    def put_json(self, key: str, obj: object) -> None:
        self.put_text(key, json.dumps(obj, indent=2, ensure_ascii=False), content_type="application/json")

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_text(self, key: str) -> str:
        return self._path(key).read_text(encoding="utf-8")

    def get_json(self, key: str) -> object:
        return json.loads(self.get_text(key))

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_prefix(self, prefix: str) -> Iterable[str]:
        base = self._path(prefix)
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                yield str(p.relative_to(self.root)).replace(os.sep, "/")

    def delete_prefix(self, prefix: str) -> None:
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def stream_writable(self, key: str, content_type: str = "application/octet-stream") -> IO[bytes]:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        return open(p, "wb")

    def stream_readable(self, key: str) -> IO[bytes]:
        return open(self._path(key), "rb")

    def local_path(self, key: str) -> Path | None:
        return self._path(key)


# ----------------------------------------------------------------------------
# GCS implementation (lazy import — only loaded if backend is "gcs")
# ----------------------------------------------------------------------------
class GCSStorage:
    def __init__(self, bucket_name: str, gcs_force_appendable: bool = False) -> None:
        from google.cloud import storage  # type: ignore

        self._bucket_name = bucket_name
        self._force_appendable = gcs_force_appendable
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        use_appendable = self._force_appendable or self._bucket_name in _APPENDABLE_BUCKETS
        if use_appendable:
            _appendable_put_sync(self._bucket_name, key, data)
            return

        blob = self._bucket.blob(key)
        try:
            blob.upload_from_string(data, content_type=content_type)
        except Exception as e:
            if _error_text_requires_appendable_bucket(e):
                log.info(
                    "GCS bucket requires appendable uploads; switching to gRPC appendable writes (%s)",
                    self._bucket_name,
                )
                _APPENDABLE_BUCKETS.add(self._bucket_name)
                _appendable_put_sync(self._bucket_name, key, data)
                return
            raise

    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> None:
        self.put_bytes(key, text.encode("utf-8"), content_type=content_type)

    def put_json(self, key: str, obj: object) -> None:
        self.put_text(
            key, json.dumps(obj, indent=2, ensure_ascii=False), content_type="application/json"
        )

    def get_bytes(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")

    def get_json(self, key: str) -> object:
        return json.loads(self.get_text(key))

    def exists(self, key: str) -> bool:
        return self._bucket.blob(key).exists(self._client)

    def list_prefix(self, prefix: str) -> Iterable[str]:
        for blob in self._client.list_blobs(self._bucket, prefix=prefix):
            yield blob.name

    def delete_prefix(self, prefix: str) -> None:
        blobs = list(self._client.list_blobs(self._bucket, prefix=prefix))
        for blob in blobs:
            blob.delete()

    def stream_writable(self, key: str, content_type: str = "application/octet-stream") -> IO[bytes]:
        blob = self._bucket.blob(key)
        return blob.open("wb", content_type=content_type)

    def stream_readable(self, key: str) -> IO[bytes]:
        return self._bucket.blob(key).open("rb")

    def local_path(self, key: str) -> Path | None:
        return None  # remote backend has no local path


# ----------------------------------------------------------------------------
# URI helpers
# ----------------------------------------------------------------------------
def storage_uri_for(storage: StorageBackend, key: str) -> str:
    """Return a stable, backend-aware URI for a stored object.

    - GCS:   gs://<bucket>/<key>
    - Local: local://<key>     (key is bucket-relative; resolves under storage_local_dir)

    The URI is what we persist in Postgres so we can round-trip back to the
    storage backend later without coupling DB rows to a specific backend.
    """
    if isinstance(storage, GCSStorage):
        return f"gs://{storage._bucket_name}/{key}"
    return f"local://{key}"


def key_from_uri(uri: str) -> str:
    """Inverse of storage_uri_for — return the bucket-relative key only."""
    if uri.startswith("gs://"):
        rest = uri[len("gs://"):]
        # Strip the bucket name → keep the key
        slash = rest.find("/")
        return rest[slash + 1:] if slash >= 0 else ""
    if uri.startswith("local://"):
        return uri[len("local://"):]
    return uri  # fallback: treat as already-key


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def get_storage(settings: Settings | None = None) -> StorageBackend:
    s = settings or get_settings()
    if s.storage_backend == "gcs":
        if not s.gcs_bucket:
            raise RuntimeError("STORAGE_BACKEND=gcs requires GCS_BUCKET")
        return GCSStorage(s.gcs_bucket, gcs_force_appendable=s.gcs_force_appendable)
    return LocalStorage(s.storage_local_dir)


# ----------------------------------------------------------------------------
# Helper: stage a remote file/prefix locally for tools that need a real path
# ----------------------------------------------------------------------------
def stage_to_tempdir(
    storage: StorageBackend,
    prefix: str,
    *,
    parallelism: int | None = None,
) -> Path:
    """Download every key under `prefix` to a temp dir; return the temp dir.

    For LocalStorage this just returns the directory directly (no copy).
    Caller is responsible for cleanup unless using LocalStorage.

    Downloads run in parallel — important on slow buckets (Rapid / HNS) where
    each per-object fetch is the bottleneck. `parallelism` defaults to
    Settings.stage_parallelism (16).
    """
    if isinstance(storage, LocalStorage):
        return storage.root / prefix

    pool_size = parallelism if parallelism is not None else max(1, get_settings().stage_parallelism)
    tmp = Path(tempfile.mkdtemp(prefix="autoeval-"))
    keys = list(storage.list_prefix(prefix))

    def _download(key: str) -> None:
        rel = key[len(prefix):].lstrip("/")
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with storage.stream_readable(key) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)

    if keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
            # list(pool.map(...)) so we surface any per-key exception.
            list(pool.map(_download, keys))
    return tmp
