"""Feature construction, fitted on training rows only.

Everything learned here (which categories are common enough to keep, the
reference month, the lookup tables used to fill in location details at
serving time) comes from the training period. Validation and test rows are
only transformed, never used to fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dxb_prices import normalise
from dxb_prices.clean import ROOM_LEVELS
from dxb_prices.config import FeatureRules

OTHER = "__other__"
MISSING = "__none__"

NUMERIC_FEATURES: tuple[str, ...] = ("area_sqm", "is_off_plan", "is_freehold", "month_index")
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "community",
    "project",
    "rooms",
    "sub_type",
    "nearest_metro",
    "nearest_mall",
    "nearest_landmark",
)
FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
POI_COLUMNS: tuple[str, ...] = ("nearest_metro", "nearest_mall", "nearest_landmark")
SUB_TYPES: tuple[str, ...] = ("Flat", "Hotel Apartment")


@dataclass
class FeatureSpec:
    """Everything needed to rebuild features for new rows. JSON-serialisable."""

    reference_month: str
    levels: dict[str, list[str]]
    community_rows: dict[str, int]
    project_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    community_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    community_names: dict[str, str] = field(default_factory=dict)
    project_names: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        return cls(**data)


def month_number(values: pd.Series) -> pd.Series:
    """Months since year 0, from dates or "YYYY-MM" strings."""
    periods = pd.to_datetime(values).dt.to_period("M")
    number: pd.Series = (periods.dt.year * 12 + periods.dt.month - 1).astype("int64")
    return number


def month_index(values: pd.Series, reference_month: str) -> pd.Series:
    ref = month_number(pd.Series([reference_month]))[0]
    index: pd.Series = (month_number(values) - ref).astype("float64")
    return index


def _levels(values: pd.Series, min_rows: int) -> list[str]:
    counts = values.dropna().value_counts()
    kept = sorted(str(v) for v in counts[counts >= min_rows].index)
    return [*kept, OTHER, MISSING]


def _mode(values: pd.Series) -> Any:
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    top = counts[counts == counts.iloc[0]].index
    value = sorted(top, key=str)[0]
    return bool(value) if isinstance(value, (bool, np.bool_)) else str(value)


def _context(train: pd.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    cols = [*POI_COLUMNS, "is_freehold"] + (["community"] if key == "project" else [])
    out: dict[str, dict[str, Any]] = {}
    for name, part in train.dropna(subset=[key]).groupby(key):
        out[str(name)] = {c: _mode(part[c]) for c in cols}
    return out


def fit(train: pd.DataFrame, rules: FeatureRules) -> FeatureSpec:
    if train.empty:
        raise ValueError("cannot fit features on an empty training set")
    levels = {
        "community": _levels(train["community"], rules.min_rows_community),
        "project": _levels(train["project"], rules.min_rows_project),
        "rooms": [*ROOM_LEVELS, OTHER, MISSING],
        "sub_type": [*SUB_TYPES, OTHER, MISSING],
    }
    for col in POI_COLUMNS:
        levels[col] = _levels(train[col], rules.min_rows_poi)
    return FeatureSpec(
        reference_month=str(train["month"].min()),
        levels=levels,
        community_rows={str(k): int(v) for k, v in train["community"].value_counts().items()},
        project_context=_context(train, "project"),
        community_context=_context(train, "community"),
        # Every English and Arabic spelling seen in training resolves at serving time.
        community_names=normalise.lookup_table(
            train["community"],
            train.get("community_en_raw", train["community"]),
            train.get("community_ar"),
        ),
        project_names=normalise.lookup_table(train["project"], train["project"], None),
    )


def _categorical(values: pd.Series, levels: list[str]) -> pd.Series:
    known = set(levels)
    mapped = [
        MISSING
        if (v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)))
        else (str(v) if str(v) in known else OTHER)
        for v in values.astype(object)
    ]
    return pd.Series(pd.Categorical(mapped, categories=levels), index=values.index)


def transform(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the model matrix. Unseen or rare categories become OTHER; missing become MISSING."""
    missing = {"area_sqm", "is_off_plan", "is_freehold", "month", *CATEGORICAL_FEATURES} - set(
        frame.columns
    )
    if missing:
        raise KeyError(f"frame is missing columns: {sorted(missing)}")
    x = pd.DataFrame(index=frame.index)
    x["area_sqm"] = frame["area_sqm"].astype("float64")
    x["is_off_plan"] = frame["is_off_plan"].astype("Float64").astype("float64")
    x["is_freehold"] = frame["is_freehold"].astype("Float64").astype("float64")
    x["month_index"] = month_index(frame["month"], spec.reference_month)
    for col in CATEGORICAL_FEATURES:
        x[col] = _categorical(frame[col], spec.levels[col])
    return x[list(FEATURES)]
