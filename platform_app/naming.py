"""Filename sanitization shared between the API (Content-Disposition + zip entry
paths) and the pipeline (Explorer HTML column in summary spreadsheets).

Kept tiny on purpose — single helper, no dependencies.
"""

from __future__ import annotations

import re


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, *, fallback: str = "episode") -> str:
    """Sanitize a name for use in Content-Disposition or zip-entry paths.

    Strips path separators and any non-[A-Za-z0-9._-] runs, collapsing to "_".
    Falls back to `fallback` if the result is empty.
    """
    cleaned = _FILENAME_SAFE.sub("_", name).strip("._-")
    return cleaned or fallback
