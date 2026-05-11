"""FastAPI entrypoint. API-only. The frontend SPA is a separate service."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(title="Autoeval Platform API", version="0.1.0")
    settings = get_settings()

    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_router)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "autoeval-api",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
