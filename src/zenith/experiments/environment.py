"""Capture the environment of a run, for reproducibility.

Design Principles
-----------------
A run should be able to explain itself later: what versions, what seed, what
commit. This captures a small, JSON-serializable snapshot with no hard
dependency beyond the standard library (torch/git are best-effort).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

__all__ = ["capture_environment"]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:  # pragma: no cover - git may be absent
        return None


def capture_environment(seed: int | None = None) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the current run environment."""
    env: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": seed,
        "git_commit": _git_commit(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        pass
    return env
