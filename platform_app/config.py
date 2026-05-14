"""Env-driven settings. Locally backed by .env; on Cloud Run by --set-env-vars/--set-secrets."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Push .env values into os.environ so libraries that read env vars directly
# (notably google-auth via GOOGLE_APPLICATION_CREDENTIALS) can see them.
# pydantic-settings loads .env into our Settings object but does NOT mutate
# os.environ, so we need this in addition.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-5.1"
    openai_max_tokens: int = 16000

    # Storage
    storage_backend: str = "local"  # "local" | "gcs"
    storage_local_dir: str = "./storage"
    gcs_bucket: str = ""
    # Rapid/zonal buckets reject JSON multipart uploads; use gRPC appendable writes instead.
    # If unset, the backend retries with appendable uploads after the first 400 from GCS.
    gcs_force_appendable: bool = False

    # Parallelism — how many episodes evaluate concurrently inside a single run.
    # Each thread holds one OpenAI request at a time. Tune down if you hit TPM caps.
    max_parallel_episodes: int = 10
    # Parallelism for ingest writes (one bucket object per write). Higher = faster
    # zip → bucket extraction on slow buckets (Rapid / HNS).
    ingest_parallelism: int = 16
    # Parallelism for staging the run prefix back from the bucket (used by the
    # explorer step). Higher = faster end-of-run packaging on slow buckets.
    stage_parallelism: int = 16

    # Explorer — self-contained HTML bundle written at end of each run.
    # Path is resolved relative to project_root() (i.e. backend/).
    explorer_template_path: str = "templates/explorer_template.html"
    # Files larger than this are replaced with a placeholder in the explorer.
    explorer_max_file_bytes: int = 5_000_000

    # Postgres state layer
    # Async URL (postgresql+asyncpg://...) used by FastAPI routes.
    # Sync URL (postgresql+psycopg://...) used by the background worker and Alembic.
    database_url: str = ""
    database_url_sync: str = ""
    db_pool_size: int = 5
    db_max_overflow: int = 10
    # Episodes left "running" longer than this are considered stale (worker crash).
    episode_stale_after_seconds: int = 900

    # Auth
    api_token: str = ""  # empty -> open access

    # CORS (dev only)
    cors_allow_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
