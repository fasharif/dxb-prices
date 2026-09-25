"""Map the two known source layouts onto one canonical column set.

* ``dld_export``: the CSV export of the DLD open data page (upper-case names).
* ``dubai_data``: the Dubai Pulse / data.dubai layout (lower-case names).

Price-per-area columns are dropped here, before anything else sees them: the
model's target is price, and price per square metre is that target divided
by the size, so keeping it would let the answer leak into the features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Columns requested from the DLD export, in this order. The export returns
# exactly the columns named in the request's "labels".
EXPORT_COLUMNS: tuple[str, ...] = (
    "TRANSACTION_NUMBER",
    "INSTANCE_DATE",
    "GROUP_EN",
    "PROCEDURE_EN",
    "IS_OFFPLAN_EN",
    "IS_FREE_HOLD_EN",
    "USAGE_EN",
    "AREA_EN",
    "AREA_AR",
    "PROP_TYPE_EN",
    "PROP_SB_TYPE_EN",
    "TRANS_VALUE",
    "PROCEDURE_AREA",
    "ACTUAL_AREA",
    "ROOMS_EN",
    "PARKING",
    "NEAREST_METRO_EN",
    "NEAREST_MALL_EN",
    "NEAREST_LANDMARK_EN",
    "TOTAL_BUYER",
    "TOTAL_SELLER",
    "MASTER_PROJECT_EN",
    "MASTER_PROJECT_AR",
    "PROJECT_EN",
    "PROJECT_AR",
)

# Columns that encode the price we are trying to predict.
LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"meter_sale_price", "meter_rent_price", "rent_value", "METER_SALE_PRICE", "PRICE_PER_METER"}
)

CANONICAL_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "transaction_date",
    "group",
    "procedure",
    "is_off_plan",
    "is_freehold",
    "usage",
    "community_en",
    "community_ar",
    "property_type",
    "property_sub_type",
    "price_aed",
    "area_sqm",
    "procedure_area_sqm",
    "rooms_raw",
    "parking",
    "nearest_metro",
    "nearest_mall",
    "nearest_landmark",
    "master_project_en",
    "master_project_ar",
    "project_en",
    "project_ar",
)

DLD_EXPORT_MAP: dict[str, str] = {
    "TRANSACTION_NUMBER": "transaction_id",
    "INSTANCE_DATE": "transaction_date",
    "GROUP_EN": "group",
    "PROCEDURE_EN": "procedure",
    "IS_OFFPLAN_EN": "is_off_plan",
    "IS_FREE_HOLD_EN": "is_freehold",
    "USAGE_EN": "usage",
    "AREA_EN": "community_en",
    "AREA_AR": "community_ar",
    "PROP_TYPE_EN": "property_type",
    "PROP_SB_TYPE_EN": "property_sub_type",
    "TRANS_VALUE": "price_aed",
    "ACTUAL_AREA": "area_sqm",
    "PROCEDURE_AREA": "procedure_area_sqm",
    "ROOMS_EN": "rooms_raw",
    "PARKING": "parking",
    "NEAREST_METRO_EN": "nearest_metro",
    "NEAREST_MALL_EN": "nearest_mall",
    "NEAREST_LANDMARK_EN": "nearest_landmark",
    "MASTER_PROJECT_EN": "master_project_en",
    "MASTER_PROJECT_AR": "master_project_ar",
    "PROJECT_EN": "project_en",
    "PROJECT_AR": "project_ar",
}

DUBAI_DATA_MAP: dict[str, str] = {
    "transaction_id": "transaction_id",
    "instance_date": "transaction_date",
    "trans_group_en": "group",
    "procedure_name_en": "procedure",
    "reg_type_en": "is_off_plan",
    "usage_en": "usage",
    "property_usage_en": "usage",
    "area_name_en": "community_en",
    "area_name_ar": "community_ar",
    "property_type_en": "property_type",
    "property_sub_type_en": "property_sub_type",
    "actual_worth": "price_aed",
    "procedure_area": "area_sqm",
    "rooms_en": "rooms_raw",
    "has_parking": "parking",
    "nearest_metro_en": "nearest_metro",
    "nearest_mall_en": "nearest_mall",
    "nearest_landmark_en": "nearest_landmark",
    "master_project_en": "master_project_en",
    "master_project_ar": "master_project_ar",
    "project_name_en": "project_en",
    "project_name_ar": "project_ar",
}

_OFF_PLAN_VALUES = {"off-plan": True, "off-plan properties": True, "off plan": True}
_READY_VALUES = {"ready": False, "existing properties": False, "existing": False}
_FREEHOLD_VALUES = {
    "free hold": True,
    "freehold": True,
    "non free hold": False,
    "non freehold": False,
}


class SchemaError(ValueError):
    """The input does not match any known source layout."""


def detect_layout(columns: pd.Index) -> str:
    cols = set(columns)
    if {"TRANSACTION_NUMBER", "TRANS_VALUE"} <= cols:
        return "dld_export"
    if {"transaction_id", "actual_worth"} <= cols:
        return "dubai_data"
    raise SchemaError(
        "unrecognised columns; expected the DLD export (TRANSACTION_NUMBER, TRANS_VALUE, ...) "
        f"or the Dubai Pulse layout (transaction_id, actual_worth, ...); got {sorted(cols)[:12]}"
    )


def _to_bool(series: pd.Series, mapping: dict[str, bool]) -> pd.Series:
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.map(mapping).astype("boolean")


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename, drop leaky columns and coerce types. Rows are not filtered here."""
    layout = detect_layout(raw.columns)
    leaky = sorted(set(raw.columns) & LEAKY_COLUMNS)
    if leaky:
        log.info("dropping price-per-area columns before any processing: %s", leaky)
    df = raw.drop(columns=leaky)
    mapping = DLD_EXPORT_MAP if layout == "dld_export" else DUBAI_DATA_MAP
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[list(CANONICAL_COLUMNS)].copy()

    df["transaction_id"] = df["transaction_id"].astype("string").str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    for col in ("price_aed", "area_sqm", "procedure_area_sqm", "parking"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["is_off_plan"] = _to_bool(df["is_off_plan"], _OFF_PLAN_VALUES | _READY_VALUES)
    df["is_freehold"] = _to_bool(df["is_freehold"], _FREEHOLD_VALUES)
    text_cols = [
        c
        for c in CANONICAL_COLUMNS
        if c
        not in {
            "transaction_id",
            "transaction_date",
            "price_aed",
            "area_sqm",
            "procedure_area_sqm",
            "parking",
            "is_off_plan",
            "is_freehold",
        }
    ]
    for col in text_cols:
        values = df[col].astype("string").str.strip()
        df[col] = values.mask(values.isin(["", "NA", "N/A", "null", "None", "nan"]))
    return df


def read_raw_csvs(paths: list[Path]) -> pd.DataFrame:
    """Read one or more raw CSV files (UTF-8, with or without a BOM) into one frame."""
    if not paths:
        raise FileNotFoundError(
            "no raw CSV files found; run `dxb-prices download` first or place DLD exports "
            "in data/raw/dld/"
        )
    frames = [pd.read_csv(p, dtype=str, encoding="utf-8-sig", keep_default_na=False) for p in paths]
    return pd.concat(frames, ignore_index=True)
