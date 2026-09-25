"""Turn canonical raw transactions into one row per apartment sale.

Each step is recorded with the number of rows it removed, so the effect of
every rule can be reported (see ``CleaningReport.to_markdown``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dxb_prices import normalise
from dxb_prices.config import CleaningRules
from dxb_prices.schema import LEAKY_COLUMNS

ROOM_LEVELS: tuple[str, ...] = ("studio", "1", "2", "3", "4", "5+", "penthouse", "unknown")
_BEDROOMS = re.compile(r"^\s*(\d+)\s*b\s*/?\s*r", re.IGNORECASE)

CLEAN_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "transaction_date",
    "month",
    "procedure",
    "is_off_plan",
    "is_freehold",
    "community",
    "community_en_raw",
    "community_ar",
    "project",
    "sub_type",
    "rooms",
    "area_sqm",
    "price_aed",
    "nearest_metro",
    "nearest_mall",
    "nearest_landmark",
)


@dataclass
class CleaningStep:
    name: str
    rule: str
    rows_before: int
    rows_after: int

    @property
    def removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class CleaningReport:
    steps: list[CleaningStep] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["| Step | Rule | Rows in | Removed | Rows out |", "|---|---|---:|---:|---:|"]
        for s in self.steps:
            lines.append(
                f"| {s.name} | {s.rule} | {s.rows_before:,} | {s.removed:,} | {s.rows_after:,} |"
            )
        return "\n".join(lines)

    def as_records(self) -> list[dict[str, object]]:
        return [
            {
                "step": s.name,
                "rule": s.rule,
                "rows_in": s.rows_before,
                "removed": s.removed,
                "rows_out": s.rows_after,
            }
            for s in self.steps
        ]


def parse_rooms(value: object) -> str:
    """Map DLD room labels ("Studio", "2 B/R", "PENTHOUSE", "NA") to a small fixed set."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = value.strip().casefold()
    if text == "studio":
        return "studio"
    if "penthouse" in text:
        return "penthouse"
    match = _BEDROOMS.match(text)
    if match:
        n = int(match.group(1))
        return "5+" if n >= 5 else str(n) if n >= 1 else "studio"
    return "unknown"


def _proc_key(value: str) -> str:
    return normalise.key_en(value)


class _Tracker:
    def __init__(self, report: CleaningReport) -> None:
        self.report = report

    def apply(
        self, df: pd.DataFrame, name: str, rule: str, fn: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        before = len(df)
        out = fn(df)
        self.report.steps.append(CleaningStep(name, rule, before, len(out)))
        return out


_LOCATION_LABELS = ("nearest_metro", "nearest_mall", "nearest_landmark")


def _drop_location_duplicates(d: pd.DataFrame) -> pd.DataFrame:
    """Keep one row where rows differ only in their nearest metro, mall or landmark.

    The export sometimes lists one sale twice, once per matching station. The
    row with the alphabetically first labels is kept, so the result does not
    depend on file order.
    """
    identity = [c for c in d.columns if c not in _LOCATION_LABELS]
    ordered = d.sort_values(list(_LOCATION_LABELS), na_position="last", kind="stable")
    kept = ordered.drop_duplicates(subset=identity, keep="first")
    return d.loc[d.index.isin(kept.index)]


def clean(canonical: pd.DataFrame, rules: CleaningRules) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the fixed validity rules. Input must come from ``schema.to_canonical``."""
    leaked = set(canonical.columns) & LEAKY_COLUMNS
    if leaked:
        raise ValueError(f"price-per-area columns must be dropped first: {sorted(leaked)}")
    report = CleaningReport()
    t = _Tracker(report)
    df = canonical.copy()
    df["area_sqm"] = df["area_sqm"].fillna(df["procedure_area_sqm"])

    df = t.apply(
        df,
        "Parse",
        "drop rows without an id, date, price or size",
        lambda d: d.dropna(subset=["transaction_id", "transaction_date", "price_aed", "area_sqm"]),
    )
    df = t.apply(
        df, "Exact duplicates", "keep one copy of identical rows", lambda d: d.drop_duplicates()
    )
    # Filter to sales before looking at repeated transaction numbers: lease-to-own
    # contracts are listed under the same number in both the Sales and Mortgage groups.
    df = t.apply(
        df,
        "Sales only",
        "keep the Sales group (drop mortgages and gifts)",
        lambda d: d.loc[d["group"].str.casefold().eq("sales").fillna(False)],
    )
    df = t.apply(
        df,
        "Location duplicates",
        "one sale listed twice with different nearest metro/mall/landmark labels: keep one",
        _drop_location_duplicates,
    )

    def drop_multi_unit(d: pd.DataFrame) -> pd.DataFrame:
        repeated = d["transaction_id"].duplicated(keep=False)
        return d.loc[~repeated]

    df = t.apply(
        df,
        "Multi-unit deals",
        "drop transaction numbers still on several rows (one value for several units)",
        drop_multi_unit,
    )

    allowed = {_proc_key(p) for p in rules.market_sale_procedures}
    df = t.apply(
        df,
        "Market sales",
        "keep sale procedures (freehold and non-freehold); drop lease-to-own",
        lambda d: d.loc[d["procedure"].map(_proc_key, na_action="ignore").isin(allowed)],
    )

    sub_types = {s.casefold() for s in rules.residential_sub_types}

    def residential_units(d: pd.DataFrame) -> pd.DataFrame:
        is_unit = d["property_type"].str.casefold().eq("unit").fillna(False)
        is_res = d["usage"].str.casefold().eq("residential").fillna(False)
        is_flat = d["property_sub_type"].str.casefold().isin(sub_types).fillna(False)
        return d.loc[is_unit & is_res & is_flat]

    df = t.apply(
        df,
        "Apartments",
        "units with residential usage, sub-type Flat or Hotel Apartment",
        residential_units,
    )
    df = t.apply(
        df,
        "Size bounds",
        f"{rules.min_area_sqm:g} to {rules.max_area_sqm:,.0f} sqm",
        lambda d: d.loc[d["area_sqm"].between(rules.min_area_sqm, rules.max_area_sqm)],
    )
    df = t.apply(
        df,
        "Price bounds",
        f"{rules.min_price_aed:,.0f} to {rules.max_price_aed:,.0f} AED",
        lambda d: d.loc[d["price_aed"].between(rules.min_price_aed, rules.max_price_aed)],
    )
    df = t.apply(
        df,
        "Price per sqm bounds",
        f"{rules.min_price_per_sqm:,.0f} to {rules.max_price_per_sqm:,.0f} AED/sqm",
        lambda d: d.loc[
            (d["price_aed"] / d["area_sqm"]).between(
                rules.min_price_per_sqm, rules.max_price_per_sqm
            )
        ],
    )

    out = pd.DataFrame(index=df.index)
    out["transaction_id"] = df["transaction_id"]
    out["transaction_date"] = df["transaction_date"]
    out["month"] = df["transaction_date"].dt.strftime("%Y-%m")
    out["procedure"] = df["procedure"]
    out["is_off_plan"] = df["is_off_plan"]
    out["is_freehold"] = df["is_freehold"]
    out["community"] = normalise.canonical_english(df["community_en"], df["community_ar"])
    out["community_en_raw"] = df["community_en"]
    out["community_ar"] = df["community_ar"]
    out["project"] = normalise.canonical_english(df["project_en"])
    out["sub_type"] = df["property_sub_type"].map(normalise.display_en, na_action="ignore")
    out["rooms"] = df["rooms_raw"].map(parse_rooms)
    out["area_sqm"] = df["area_sqm"].astype(np.float64)
    out["price_aed"] = df["price_aed"].astype(np.float64)
    for col in ("nearest_metro", "nearest_mall", "nearest_landmark"):
        out[col] = df[col].map(normalise.display_en, na_action="ignore").astype("string")
    out = t.apply(
        out,
        "Community known",
        "drop rows without a community name",
        lambda d: d.dropna(subset=["community"]),
    )
    out = out.sort_values(["transaction_date", "transaction_id"], kind="stable")
    return out.reset_index(drop=True)[list(CLEAN_COLUMNS)], report
