"""Model rows from what a caller supplies, built the same way for the API and the evaluation.

A caller gives the community, size, rooms, off-plan status and date, and
optionally the project, sub-type and freehold status. Nobody types in DLD's
nearest metro, mall or landmark, so those come from the training rows: the
project's most common values when the project is known, otherwise the
community's. A missing freehold flag is filled the same way.

Each row is then routed to one of the two models:

* ``full`` when the project is in the training data;
* ``community`` when it is not given or not in the training data. That model
  was trained without the project and the location labels, so it does not
  have to guess them.

The evaluation scores the test month through these same functions, so the
published accuracy is the accuracy of what the API returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from dxb_prices.features import COMMUNITY, FULL, POI_COLUMNS, FeatureSpec
from dxb_prices.model import PriceModel

FloatArray = npt.NDArray[np.float64]

REQUEST_COLUMNS: tuple[str, ...] = (
    "community",
    "project",
    "rooms",
    "sub_type",
    "area_sqm",
    "is_off_plan",
    "is_freehold",
    "month",
)


def _context_column(context: dict[str, dict[str, Any]], keys: pd.Series, column: str) -> pd.Series:
    lookup = {k: v.get(column) for k, v in context.items()}
    return keys.map(lambda k: lookup.get(k) if isinstance(k, str) else None)


def model_rows(requests: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Fill in what callers do not supply and choose each row's model ("variant" column)."""
    missing = set(REQUEST_COLUMNS) - set(requests.columns)
    if missing:
        raise KeyError(f"requests are missing columns: {sorted(missing)}")
    out = requests[list(REQUEST_COLUMNS)].copy()
    project = out["project"].astype("object")
    known = project.map(lambda p: isinstance(p, str) and p in spec.project_context).astype(bool)
    project = project.where(known, None)
    community = out["community"].astype("object")
    for col in (*POI_COLUMNS, "is_freehold"):
        from_project = _context_column(spec.project_context, project, col)
        from_community = _context_column(spec.community_context, community, col)
        filled = from_project.where(from_project.notna(), from_community)
        if col == "is_freehold":
            given = out["is_freehold"].astype("object")
            filled = given.where(given.notna(), filled)
            out[col] = pd.array(
                [None if pd.isna(v) else bool(v) for v in filled],
                dtype="boolean",
            )
        else:
            out[col] = filled.astype("string")
    out["project"] = project
    out["variant"] = np.where(known, FULL, COMMUNITY)
    return out


def requests_from_sales(sales: pd.DataFrame, *, with_project: bool) -> pd.DataFrame:
    """What a caller would send for each recorded sale: no location labels, no freehold flag."""
    requests = sales[[c for c in REQUEST_COLUMNS if c != "is_freehold"]].copy()
    requests["is_freehold"] = pd.array([pd.NA] * len(sales), dtype="boolean")
    if not with_project:
        requests["project"] = None
    return requests[list(REQUEST_COLUMNS)]


def estimate(model: PriceModel, rows: pd.DataFrame, *, with_range: bool = True) -> pd.DataFrame:
    """Price estimate (and 80% range) for rows from ``model_rows``, each from its own variant."""
    out = pd.DataFrame(index=rows.index, columns=["estimate", "low", "high"], dtype="float64")
    for variant in (FULL, COMMUNITY):
        part = rows.loc[rows["variant"] == variant]
        if part.empty:
            continue
        out.loc[part.index, "estimate"] = model.predict(part, variant)
        if with_range:
            low, high = model.interval(part, variant)
            out.loc[part.index, "low"] = low
            out.loc[part.index, "high"] = high
    return out
