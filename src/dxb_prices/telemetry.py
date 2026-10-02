"""Switch off third-party usage telemetry before those libraries are imported.

MLflow 3 decides whether to start its telemetry client while ``mlflow`` is
being imported, and Evidently checks ``DO_NOT_TRACK`` before each event. The
opt-out therefore has to be in the environment before either is imported:
``tracking`` and ``drift`` call :func:`opt_out` when they are loaded, and so
does the command line entry point. A value the user has already set is kept.
"""

from __future__ import annotations

import os

OPT_OUT: dict[str, str] = {
    # MLflow 3: read by mlflow.telemetry.utils.is_telemetry_disabled().
    "MLFLOW_DISABLE_TELEMETRY": "true",
    # Evidently (any value) and MLflow ("true") both honour DO_NOT_TRACK.
    "DO_NOT_TRACK": "true",
    # Read by Evidently's legacy module.
    "EVIDENTLY_DISABLE_TELEMETRY": "1",
}


def opt_out() -> None:
    for name, value in OPT_OUT.items():
        os.environ.setdefault(name, value)
