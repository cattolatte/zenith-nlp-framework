"""MLflow experiment tracking for Zenith.

Design Principles
-----------------
Tracking is optional and lazily wired: ``mlflow`` and ``omegaconf`` are imported
only when a real tracker is constructed, so ``import zenith`` never drags them in
and Zenith runs fine without them. Call sites stay branch-free by using
:class:`NoOpTracker` when tracking is disabled.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = ["MlflowTracker", "NoOpTracker", "flatten_config", "get_tracker"]


def flatten_config(config: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict / OmegaConf config into ``{"a.b.c": value}`` pairs.

    MLflow params are flat scalars, so a nested (Hydra) config must be flattened
    before logging. OmegaConf containers are converted via a lazy import.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass

    items: dict[str, Any] = {}
    if isinstance(config, Mapping):
        for key, value in config.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, (Mapping, list, tuple)):
                items.update(flatten_config(value, prefix=f"{new_key}."))
            else:
                items[new_key] = value
    elif isinstance(config, (list, tuple)):
        for idx, value in enumerate(config):
            new_key = f"{prefix}{idx}"
            if isinstance(value, (Mapping, list, tuple)):
                items.update(flatten_config(value, prefix=f"{new_key}."))
            else:
                items[new_key] = value
    return items


class MlflowTracker:
    """A thin wrapper over the MLflow fluent API.

    Parameters
    ----------
    experiment : str
        Experiment name to log under.
    tracking_uri : str, optional
        Tracking-server URI (e.g. ``http://localhost:5000``); defaults to local
        ``./mlruns``.
    """

    def __init__(self, experiment: str = "zenith", tracking_uri: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(
        self, run_name: str | None = None, tags: Mapping[str, str] | None = None
    ) -> Iterator["MlflowTracker"]:
        """Open (and always close) an MLflow run."""
        with self._mlflow.start_run(run_name=run_name, tags=dict(tags) if tags else None):
            yield self

    def log_config(self, config: Any) -> None:
        """Flatten and log a (Hydra/OmegaConf) config as MLflow params."""
        self.log_params(flatten_config(config))

    def log_params(self, params: Mapping[str, Any]) -> None:
        safe = {k: str(v)[:250] for k, v in params.items()}
        if safe:
            self._mlflow.log_params(safe)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for name, value in metrics.items():
            self._mlflow.log_metric(name, float(value), step=step)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log a text blob (e.g. a generated sample) as a run artifact."""
        self._mlflow.log_text(text, artifact_file)

    def log_artifact(self, path: str | Path) -> None:
        """Log a file (e.g. a checkpoint) as a run artifact."""
        self._mlflow.log_artifact(str(path))


class NoOpTracker:
    """Null tracker; every method is a no-op so call sites need no branching."""

    @contextmanager
    def run(self, *args: Any, **kwargs: Any) -> Iterator["NoOpTracker"]:
        yield self

    def log_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_params(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_text(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None


def get_tracker(
    *, enabled: bool = True, experiment: str = "zenith", tracking_uri: str | None = None
) -> MlflowTracker | NoOpTracker:
    """Return a real tracker when enabled, else a no-op one."""
    if enabled:
        return MlflowTracker(experiment=experiment, tracking_uri=tracking_uri)
    return NoOpTracker()
