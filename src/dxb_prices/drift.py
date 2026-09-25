"""Evidently data drift report: the newest month against the training period."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_COLUMNS: tuple[str, ...] = ("area_sqm", "log_price_per_sqm")
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "community",
    "rooms",
    "is_off_plan",
    "is_freehold",
    "sub_type",
)


def drift_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Columns compared by the report. Price per sqm is included as target drift."""
    out = pd.DataFrame(index=frame.index)
    out["area_sqm"] = frame["area_sqm"].astype("float64")
    out["log_price_per_sqm"] = np.log(frame["price_aed"] / frame["area_sqm"]).astype("float64")
    for col in CATEGORICAL_COLUMNS:
        out[col] = frame[col].astype("string").fillna("missing").astype(str)
    return out


def _parse(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    share: float | None = None
    count: float | None = None
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        kind = str(config.get("type", ""))
        if kind.endswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
            count = float(metric["value"]["count"])
        elif kind.endswith("ValueDrift"):
            method = str(config.get("method", ""))
            threshold = float(config.get("threshold", 0.0))
            value = float(metric["value"])
            drifted = value < threshold if "p_value" in method else value >= threshold
            columns.append(
                {
                    "column": config.get("column"),
                    "method": method,
                    "value": value,
                    "threshold": threshold,
                    "drifted": drifted,
                }
            )
    return {"drifted_share": share, "drifted_count": count, "columns": columns}


def run(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None) -> dict[str, Any]:
    """Compare ``current`` with ``reference``; write HTML if a path is given; return a summary."""
    # Evidently reads this when first imported; it sends usage telemetry otherwise.
    os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    if reference.empty or current.empty:
        raise ValueError("drift report needs non-empty reference and current data")
    definition = DataDefinition(
        numerical_columns=list(NUMERIC_COLUMNS), categorical_columns=list(CATEGORICAL_COLUMNS)
    )
    ref = Dataset.from_pandas(drift_frame(reference), data_definition=definition)
    cur = Dataset.from_pandas(drift_frame(current), data_definition=definition)
    with warnings.catch_warnings():
        # scipy warns on categories that are absent from one side; the test still runs.
        warnings.simplefilter("ignore", RuntimeWarning)
        snapshot = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    summary = _parse(snapshot.dict())
    summary["reference_rows"] = len(reference)
    summary["current_rows"] = len(current)
    return summary
