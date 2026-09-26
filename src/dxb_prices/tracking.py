"""MLflow experiment tracking with a local file store.

MLflow 3.x keeps the file backend in maintenance mode and refuses it unless
``MLFLOW_ALLOW_FILE_STORE=true`` is set. This project opts in for ``file:``
URIs because a plain directory is the simplest store to inspect, copy and
upload as a CI artefact. Set ``MLFLOW_TRACKING_URI=sqlite:///mlflow.db`` to
use a database store instead; nothing else changes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dxb_prices import config, telemetry

# Before anything below imports mlflow (see dxb_prices.telemetry).
telemetry.opt_out()

EXPERIMENT = "dxb-prices"


def resolve_uri(uri: str | None = None) -> str:
    chosen = (
        uri or os.environ.get("MLFLOW_TRACKING_URI") or (config.PROJECT_ROOT / "mlruns").as_uri()
    )
    if chosen.startswith("file:"):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    # MLflow records the Git commit when it can; without a git binary (slim
    # containers) GitPython prints a long warning instead. Keep it quiet.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    return chosen


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested dicts into dotted MLflow parameter names."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out |= flatten(f"{prefix}.{k}" if prefix else str(k), v)
        return out
    return {prefix: value}


@contextmanager
def run(uri: str | None, run_name: str, tags: dict[str, str]) -> Iterator[Any]:
    import mlflow

    mlflow.set_tracking_uri(resolve_uri(uri))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name, tags=tags) as active:
        yield active


def log_params(params: Mapping[str, Any]) -> None:
    import mlflow

    flat = flatten("", params)
    # MLflow limits parameter values to 6,000 characters.
    mlflow.log_params({k: str(v)[:6000] for k, v in flat.items()})


def log_metrics(metrics: Mapping[str, float]) -> None:
    import mlflow

    mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


@contextmanager
def child_run(run_name: str) -> Iterator[Any]:
    import mlflow

    with mlflow.start_run(run_name=run_name, nested=True) as active:
        yield active


def log_artifacts(directory: Path, artifact_path: str) -> None:
    import mlflow

    mlflow.log_artifacts(str(directory), artifact_path=artifact_path)
