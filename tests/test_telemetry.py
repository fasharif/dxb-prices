"""MLflow and Evidently telemetry is off as soon as this package's modules load them."""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Only what the interpreter needs; in particular none of the CI or pytest
# variables that make MLflow switch telemetry off by itself.
_KEEP = {"PATH", "HOME", "SYSTEMROOT", "TMP", "TEMP", "TMPDIR", "USERPROFILE", "LOCALAPPDATA"}

PROBE = """
import json, os
import dxb_prices.tracking, dxb_prices.drift
import mlflow
import evidently.telemetry as evidently_telemetry
from mlflow.telemetry import utils
from mlflow.telemetry.client import get_telemetry_client
print(json.dumps({
    "ci_detected": utils._IS_IN_CI_ENV_OR_TESTING,
    "mlflow_disabled": utils.is_telemetry_disabled(),
    "mlflow_client_started": get_telemetry_client() is not None,
    "evidently_do_not_track": evidently_telemetry.DO_NOT_TRACK_ENV in os.environ,
}))
"""


def test_telemetry_is_off_before_mlflow_and_evidently_load() -> None:
    env = {k: v for k, v in os.environ.items() if k in _KEEP}
    out = subprocess.run(
        [sys.executable, "-c", PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {
        # MLflow did not switch itself off because of CI; the project's opt-out did.
        "ci_detected": False,
        "mlflow_disabled": True,
        "mlflow_client_started": False,
        "evidently_do_not_track": True,
    }
