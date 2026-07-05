"""Record a training run to disk as a small set of readable files.

Design Principles
-----------------
Complementary to live MLflow tracking: a self-contained, human-readable folder per
run — ``config.json``, ``metrics.json``, ``samples.txt``, ``environment.json`` and
a combined ``run.json`` — that survives without a tracking server. Everything is
plain JSON/text so a run can be inspected or diffed with no tooling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

__all__ = ["record_run"]


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _format_samples(samples: Sequence[str]) -> str:
    return "\n\n".join(f"=== epoch {i + 1} ===\n{s}" for i, s in enumerate(samples))


def record_run(
    directory: str | Path,
    *,
    config: dict[str, Any],
    history: list[dict[str, float]],
    samples: Sequence[str] | None = None,
    environment: dict[str, Any] | None = None,
) -> Path:
    """Write a run's config, metrics, samples and environment into ``directory``.

    Returns the directory path. Creates it (and parents) if needed.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    _write_json(path / "config.json", config)
    _write_json(path / "metrics.json", {"history": history})
    if environment is not None:
        _write_json(path / "environment.json", environment)
    if samples:
        (path / "samples.txt").write_text(_format_samples(samples), encoding="utf-8")
    _write_json(
        path / "run.json",
        {
            "config": config,
            "metrics": {"history": history},
            "environment": environment,
            "num_samples": len(samples or []),
        },
    )
    return path
