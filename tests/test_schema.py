from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dxb_prices import schema
from tests.conftest import dld_rows


def test_detects_both_layouts() -> None:
    assert schema.detect_layout(pd.Index(["TRANSACTION_NUMBER", "TRANS_VALUE"])) == "dld_export"
    assert schema.detect_layout(pd.Index(["transaction_id", "actual_worth"])) == "dubai_data"
    with pytest.raises(schema.SchemaError, match="unrecognised columns"):
        schema.detect_layout(pd.Index(["id", "price"]))


def test_dld_export_maps_to_canonical_types() -> None:
    raw = dld_rows(
        {"IS_OFFPLAN_EN": "Off-Plan", "IS_FREE_HOLD_EN": "Non Free Hold", "TRANS_VALUE": "1,5"},
        {"IS_OFFPLAN_EN": "Ready", "NEAREST_METRO_EN": "NA", "PROJECT_EN": "  "},
    )
    df = schema.to_canonical(raw)
    assert list(df.columns) == list(schema.CANONICAL_COLUMNS)
    assert df["is_off_plan"].tolist() == [True, False]
    assert df["is_freehold"].tolist() == [False, True]
    assert pd.isna(df.loc[0, "price_aed"])  # "1,5" is not a number: left for cleaning to drop
    assert pd.isna(df.loc[1, "nearest_metro"]) and pd.isna(df.loc[1, "project_en"])
    assert str(df["transaction_date"].dtype).startswith("datetime64")


def test_price_per_area_columns_are_dropped_before_anything_else() -> None:
    raw = pd.DataFrame(
        [
            {
                "transaction_id": "1-11-2024-1",
                "instance_date": "2024-05-01",
                "trans_group_en": "Sales",
                "procedure_name_en": "Sell",
                "reg_type_en": "Existing Properties",
                "property_usage_en": "Residential",
                "area_name_en": "Business Bay",
                "area_name_ar": "الخليج التجاري",
                "property_type_en": "Unit",
                "property_sub_type_en": "Flat",
                "actual_worth": "1500000",
                "procedure_area": "75",
                "rooms_en": "1 B/R",
                "meter_sale_price": "20000",
                "meter_rent_price": "",
                "rent_value": "",
            }
        ]
    )
    df = schema.to_canonical(raw)
    assert not set(schema.LEAKY_COLUMNS) & set(df.columns)
    assert df.loc[0, "price_aed"] == 1_500_000
    assert df.loc[0, "area_sqm"] == 75
    assert df.loc[0, "is_off_plan"] == False  # noqa: E712 - pandas nullable boolean
    assert df.loc[0, "community_en"] == "Business Bay"


def test_read_raw_csvs_handles_bom_and_multiple_files(tmp_path: Path) -> None:
    for i in (1, 2):
        (tmp_path / f"t{i}.csv").write_bytes(
            "\ufeffTRANSACTION_NUMBER,TRANS_VALUE\n".encode() + f'"{i}","100"\n'.encode()
        )
    df = schema.read_raw_csvs(sorted(tmp_path.glob("*.csv")))
    assert list(df.columns) == ["TRANSACTION_NUMBER", "TRANS_VALUE"]
    assert df["TRANSACTION_NUMBER"].tolist() == ["1", "2"]


def test_read_raw_csvs_explains_an_empty_cache() -> None:
    with pytest.raises(FileNotFoundError, match="dxb-prices download"):
        schema.read_raw_csvs([])
