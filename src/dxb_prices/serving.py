"""Model rows from what a caller supplies, built the same way for the API and the evaluation.

A caller gives the community, size, rooms, off-plan status and date, and
optionally the project, sub-type and freehold status. Nobody types in DLD's
nearest metro, mall or landmark, so those come from the training rows: the
project's most common values when the project has training sales in that
community, otherwise the community's. A missing freehold flag is filled the
same way.

Each row is then routed to one of the two models:

* ``full`` when the project has training sales in the given community;
* ``community`` when it is not given, not in the training data, or only
  recorded in other communities. That model was trained without the project
  and the location labels, so it does not have to guess them.

The evaluation scores the test month through these same functions, so the
published accuracy is the accuracy of what the API returns.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from dxb_prices.clean import ROOM_LEVELS
from dxb_prices.features import COMMUNITY, CONTEXT_COLUMNS, FULL, FeatureSpec
from dxb_prices.model import PriceModel

FloatArray = npt.NDArray[np.float64]

# Room counts the API accepts; a sale recorded as "unknown" cannot be requested.
API_ROOMS: tuple[str, ...] = tuple(r for r in ROOM_LEVELS if r != "unknown")

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


def _project_column(
    spec: FeatureSpec, community: pd.Series, project: pd.Series, column: str
) -> pd.Series:
    values = [
        spec.project_context[c][p].get(column) if spec.has_project(c, p) else None
        for c, p in zip(community, project, strict=True)
    ]
    return pd.Series(values, index=community.index, dtype=object)


def _community_column(spec: FeatureSpec, community: pd.Series, column: str) -> pd.Series:
    lookup = {k: v.get(column) for k, v in spec.community_context.items()}
    return community.map(lambda k: lookup.get(k) if isinstance(k, str) else None)


def model_rows(requests: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Fill in what callers do not supply and choose each row's model ("variant" column)."""
    missing = set(REQUEST_COLUMNS) - set(requests.columns)
    if missing:
        raise KeyError(f"requests are missing columns: {sorted(missing)}")
    out = requests[list(REQUEST_COLUMNS)].copy()
    community = out["community"].astype("object")
    project = out["project"].astype("object")
    known = pd.Series(
        [spec.has_project(c, p) for c, p in zip(community, project, strict=True)],
        index=out.index,
        dtype=bool,
    )
    project = project.where(known, None)
    for col in CONTEXT_COLUMNS:
        from_project = _project_column(spec, community, project, col)
        from_community = _community_column(spec, community, col)
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


def refusals(sales: pd.DataFrame, spec: FeatureSpec) -> dict[str, int]:
    """How many recorded sales the API would refuse, by reason.

    The API answers 404 for a community without training sales and 422 for a
    room count it does not accept. The evaluation still scores such sales, so
    that every estimator is compared on the same rows; the report says how
    many there were.
    """
    community = sales["community"].astype("object")
    unknown = ~community.map(lambda c: isinstance(c, str) and c in spec.community_rows)
    rooms = ~sales["rooms"].astype("object").isin(API_ROOMS)
    return {
        "community without training sales (404)": int(unknown.sum()),
        "room count not recorded (422)": int((rooms & ~unknown).sum()),
    }


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
