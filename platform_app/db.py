"""SQLAlchemy engine + session factories.

Two engines share the same Postgres database:
  - async (asyncpg) — used by FastAPI route handlers.
  - sync  (psycopg) — used by the in-thread worker and Alembic.

Both are lazily constructed so importing this module is cheap and works even
when DATABASE_URL is unset (e.g. unit tests for non-DB code).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import AsyncIterator, Iterator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings, get_settings


_async_engine = None
_async_session_maker: async_sessionmaker[AsyncSession] | None = None
_sync_engine = None
_sync_session_maker: sessionmaker[Session] | None = None


def _settings() -> Settings:
    return get_settings()


def get_async_engine():
    global _async_engine, _async_session_maker
    if _async_engine is None:
        s = _settings()
        if not s.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        _async_engine = create_async_engine(
            s.database_url,
            pool_size=s.db_pool_size,
            max_overflow=s.db_max_overflow,
            pool_pre_ping=True,
            future=True,
        )
        _async_session_maker = async_sessionmaker(
            _async_engine, expire_on_commit=False, class_=AsyncSession
        )
    return _async_engine


def get_sync_engine():
    global _sync_engine, _sync_session_maker
    if _sync_engine is None:
        s = _settings()
        if not s.database_url_sync:
            raise RuntimeError("DATABASE_URL_SYNC is not configured")
        _sync_engine = create_engine(
            s.database_url_sync,
            pool_size=s.db_pool_size,
            max_overflow=s.db_max_overflow,
            pool_pre_ping=True,
            future=True,
        )
        _sync_session_maker = sessionmaker(
            _sync_engine, expire_on_commit=False, class_=Session
        )
    return _sync_engine


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Yields a session; the handler controls commits.

    On unhandled exception the session is rolled back. Handlers should call
    `await db.commit()` when they want their writes durable.
    """
    get_async_engine()
    assert _async_session_maker is not None
    async with _async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@contextmanager
def sync_session() -> Iterator[Session]:
    """Context manager for the worker thread pool. Commits on exit, rolls back on exception."""
    get_sync_engine()
    assert _sync_session_maker is not None
    session = _sync_session_maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
