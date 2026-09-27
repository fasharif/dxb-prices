"""Feature construction, fitted on training rows only.

Everything learned here (which categories are common enough to keep, the
reference month, the lookup tables used to fill in location details at
serving time) comes from the training period. Validation and test rows are
only transformed, never used to fit.

There are two feature sets. The full model uses everything, including the
project and DLD's nearest metro, mall and landmark. The community-level model
leaves out the project and the location labels; it serves requests whose
project is not given or not in the training data (see ``serving``).

A project is identified by its community and its name together. Different
buildings in different communities can share a name (in the 2026 export,
"Botanica" in Dubai Marina and in Jumeirah Village Circle), so the model's
project levels, the lookup tables and the training counts all use both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
COMMUNITY_FEATURES: tuple[str, ...] = tuple(
    f for f in FEATURES if f != "project" and f not in POI_COLUMNS
)
SUB_TYPES: tuple[str, ...] = ("Flat", "Hotel Apartment")

# The two models: "full" knows the project, "community" does not.
FULL = "full"
COMMUNITY = "community"
VARIANT_FEATURES: dict[str, tuple[str, ...]] = {FULL: FEATURES, COMMUNITY: COMMUNITY_FEATURES}

# Joins community and project into the model's project level. DLD names contain
# "|" and "/", but never this control character.
PROJECT_KEY_SEP = "\x1f"


def project_key(community: str, project: str) -> str:
    """The model's level for a project: its community and name together."""
    return f"{community}{PROJECT_KEY_SEP}{project}"


def project_keys(frame: pd.DataFrame) -> pd.Series:
    """``project_key`` for every row; missing where the community or project is missing."""
    pairs = zip(frame["community"].astype(object), frame["project"].astype(object), strict=True)
    keys = [
        project_key(c, p) if isinstance(c, str) and isinstance(p, str) else None for c, p in pairs
    ]
    return pd.Series(keys, index=frame.index, dtype=object)


@dataclass
class FeatureSpec:
    """Everything needed to rebuild features for new rows. JSON-serialisable.

    ``project_context`` and ``project_rows`` are keyed by community, then by
    project name. ``project_names`` maps any spelling of a project name to its
    canonical name; the community comes from the request.
    """

    reference_month: str
    levels: dict[str, list[str]]
    community_rows: dict[str, int]
    project_context: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    community_context: dict[str, dict[str, Any]] = field(default_factory=dict)
    community_names: dict[str, str] = field(default_factory=dict)
    project_names: dict[str, str] = field(default_factory=dict)
    project_rows: dict[str, dict[str, int]] = field(default_factory=dict)
    # Training sales a project needs for a level of its own; rarer projects share OTHER.
    min_rows_project: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSpec:
        expected = {f.name for f in fields(cls)}
        nested = set(data) == expected and all(
            isinstance(counts, dict) for counts in data["project_rows"].values()
        )
        if not nested:
            raise ValueError(
                "the saved feature spec was written by an older version of dxb-prices "
                "(its project tables are not keyed by community); train the model again"
            )
        return cls(**data)

    def has_project(self, community: object, project: object) -> bool:
        """True when this project has training sales in this community."""
        return (
            isinstance(community, str)
            and isinstance(project, str)
            and project in self.project_rows.get(community, {})
        )

    def project_has_own_level(self, community: str, project: str) -> bool:
        """True when the full model learned this project itself rather than as OTHER."""
        return project_key(community, project) in self.levels["project"]


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


CONTEXT_COLUMNS: tuple[str, ...] = (*POI_COLUMNS, "is_freehold")


def _community_context(train: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, part in train.dropna(subset=["community"]).groupby("community"):
        out[str(name)] = {c: _mode(part[c]) for c in CONTEXT_COLUMNS}
    return out


def _project_tables(
    train: pd.DataFrame,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, int]]]:
    """Each (community, project)'s usual location labels and freehold flag, and its row count."""
    context: dict[str, dict[str, dict[str, Any]]] = {}
    rows: dict[str, dict[str, int]] = {}
    known = train.dropna(subset=["community", "project"])
    for (community, project), part in known.groupby(["community", "project"]):
        context.setdefault(str(community), {})[str(project)] = {
            c: _mode(part[c]) for c in CONTEXT_COLUMNS
        }
        rows.setdefault(str(community), {})[str(project)] = len(part)
    return context, rows


def fit(train: pd.DataFrame, rules: FeatureRules) -> FeatureSpec:
    if train.empty:
        raise ValueError("cannot fit features on an empty training set")
    levels = {
        "community": _levels(train["community"], rules.min_rows_community),
        "project": _levels(project_keys(train), rules.min_rows_project),
        "rooms": [*ROOM_LEVELS, OTHER, MISSING],
        "sub_type": [*SUB_TYPES, OTHER, MISSING],
    }
    for col in POI_COLUMNS:
        levels[col] = _levels(train[col], rules.min_rows_poi)
    project_context, project_rows = _project_tables(train)
    return FeatureSpec(
        reference_month=str(train["month"].min()),
        levels=levels,
        community_rows={str(k): int(v) for k, v in train["community"].value_counts().items()},
        project_context=project_context,
        community_context=_community_context(train),
        # Every English and Arabic spelling seen in training resolves at serving time.
        community_names=normalise.lookup_table(
            train["community"],
            train.get("community_en_raw", train["community"]),
            train.get("community_ar"),
        ),
        project_names=normalise.lookup_table(train["project"], train["project"], None),
        project_rows=project_rows,
        min_rows_project=rules.min_rows_project,
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


def transform(
    frame: pd.DataFrame, spec: FeatureSpec, columns: tuple[str, ...] = FEATURES
) -> pd.DataFrame:
    """Build the model matrix with ``columns`` (a subset of FEATURES, in that order).

    Unseen or rare categories become OTHER; missing ones become MISSING.
    """
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
        values = project_keys(frame) if col == "project" else frame[col]
        x[col] = _categorical(values, spec.levels[col])
    return x[list(columns)]
