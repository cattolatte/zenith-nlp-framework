"""Generation serving (FastAPI + SSE streaming) over a Zenith checkpoint."""

from __future__ import annotations

from .api import create_app, serve

__all__ = ["create_app", "serve"]
