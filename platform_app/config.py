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

    # Auth
    api_token: str = ""  # empty -> open access

    # CORS (dev only)
    cors_allow_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
