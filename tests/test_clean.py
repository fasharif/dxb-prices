from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from dxb_prices import clean, schema
from dxb_prices.config import DEFAULT_SETTINGS
from tests.conftest import dld_rows

RULES = DEFAULT_SETTINGS.cleaning


def run(raw: pd.DataFrame) -> tuple[pd.DataFrame, clean.CleaningReport]:
    return clean.clean(schema.to_canonical(raw), RULES)


def removed_by(report: clean.CleaningReport, step: str) -> int:
    return next(s.removed for s in report.steps if s.name == step)


def test_keeps_sales_only() -> None:
    out, report = run(
        dld_rows(
            {},
            {"GROUP_EN": "Mortgage", "PROCEDURE_EN": "Mortgage Registration"},
            {"GROUP_EN": "Gifts", "PROCEDURE_EN": "Grant"},
        )
    )
    assert len(out) == 1
    assert removed_by(report, "Sales only") == 2


def test_keeps_market_sale_procedures_including_non_freehold_ones() -> None:
    out, report = run(
        dld_rows(
            {"PROCEDURE_EN": "Sell - Pre registration", "IS_OFFPLAN_EN": "Off-Plan"},
            {"PROCEDURE_EN": "Sell Development", "IS_FREE_HOLD_EN": "Non Free Hold"},
            {"PROCEDURE_EN": "Lease to Own Registration"},
        )
    )
    assert sorted(out["procedure"]) == ["Sell - Pre registration", "Sell Development"]
    assert removed_by(report, "Market sales") == 1


def test_keeps_apartments_only() -> None:
    out, _ = run(
        dld_rows(
            {},
            {"PROP_SB_TYPE_EN": "Hotel Apartment"},
            {"PROP_TYPE_EN": "Building", "PROP_SB_TYPE_EN": "Villa"},
            {"PROP_TYPE_EN": "Land", "PROP_SB_TYPE_EN": "Commercial"},
            {"PROP_SB_TYPE_EN": "Office"},
            {"USAGE_EN": "Commercial"},
        )
    )
    assert sorted(out["sub_type"]) == ["Flat", "Hotel Apartment"]


def test_exact_duplicates_keep_one_copy() -> None:
    raw = dld_rows({}, {})
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    out, report = run(raw)
    assert len(out) == 2
    assert removed_by(report, "Exact duplicates") == 1


def test_one_sale_listed_with_two_metro_stations_is_kept_once() -> None:
    raw = dld_rows(
        {"TRANSACTION_NUMBER": "7-7-2026", "NEAREST_METRO_EN": "Metro B"},
        {"TRANSACTION_NUMBER": "7-7-2026", "NEAREST_METRO_EN": "Metro A"},
    )
    out, report = run(raw)
    assert len(out) == 1
    assert out.loc[0, "nearest_metro"] == "Metro A"
    assert removed_by(report, "Location duplicates") == 1
    assert removed_by(report, "Multi-unit deals") == 0


def test_lease_to_own_listed_in_two_groups_is_removed_as_lease_to_own() -> None:
    lease = {"TRANSACTION_NUMBER": "5-5-2026", "PROCEDURE_EN": "Lease to Own Registration"}
    out, report = run(dld_rows(lease, lease | {"GROUP_EN": "Mortgage"}, {}))
    assert "5-5-2026" not in set(out["transaction_id"])
    assert removed_by(report, "Sales only") == 1
    assert removed_by(report, "Multi-unit deals") == 0
    assert removed_by(report, "Market sales") == 1


def test_multi_unit_deals_are_dropped_entirely() -> None:
    raw = dld_rows(
        {"TRANSACTION_NUMBER": "9-9-2026", "ACTUAL_AREA": "60", "PROCEDURE_AREA": "60"},
        {"TRANSACTION_NUMBER": "9-9-2026", "ACTUAL_AREA": "95", "PROCEDURE_AREA": "95"},
        {},
    )
    out, report = run(raw)
    assert "9-9-2026" not in set(out["transaction_id"])
    assert removed_by(report, "Multi-unit deals") == 2


@pytest.mark.parametrize(
    ("row", "step"),
    [
        ({"TRANS_VALUE": "50000"}, "Price bounds"),
        ({"TRANS_VALUE": "600000000", "ACTUAL_AREA": "2900"}, "Price bounds"),
        ({"ACTUAL_AREA": "10", "PROCEDURE_AREA": "10"}, "Size bounds"),
        ({"ACTUAL_AREA": "3500", "PROCEDURE_AREA": "3500"}, "Size bounds"),
        ({"TRANS_VALUE": "150000", "ACTUAL_AREA": "70"}, "Price per sqm bounds"),
        ({"TRANS_VALUE": "30000000", "ACTUAL_AREA": "70"}, "Price per sqm bounds"),
        ({"TRANS_VALUE": ""}, "Parse"),
        ({"INSTANCE_DATE": "not a date"}, "Parse"),
    ],
)
def test_validity_rules(row: dict[str, str], step: str) -> None:
    out, report = run(dld_rows({}, row))
    assert len(out) == 1
    assert removed_by(report, step) == 1


def test_genuine_luxury_sales_survive() -> None:
    # 180,000 AED/sqm on a large branded penthouse is real in 2026.
    out, _ = run(dld_rows({"TRANS_VALUE": "171000000", "ACTUAL_AREA": "931", "ROOMS_EN": "5 B/R"}))
    assert len(out) == 1


def test_actual_area_falls_back_to_procedure_area() -> None:
    out, _ = run(dld_rows({"ACTUAL_AREA": "", "PROCEDURE_AREA": "80"}))
    assert out.loc[0, "area_sqm"] == 80


def test_refuses_input_with_price_per_area_columns() -> None:
    frame = schema.to_canonical(dld_rows({})).assign(meter_sale_price=20000.0)
    with pytest.raises(ValueError, match="price-per-area"):
        clean.clean(frame, RULES)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Studio", "studio"),
        ("1 B/R", "1"),
        ("4 B/R", "4"),
        ("5 B/R", "5+"),
        ("9 B/R", "5+"),
        ("PENTHOUSE", "penthouse"),
        ("NA", "unknown"),
        ("Office", "unknown"),
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_parse_rooms(raw: str | None, expected: str) -> None:
    assert clean.parse_rooms(raw) == expected


def test_names_are_normalised() -> None:
    out, _ = run(
        dld_rows(
            {
                "AREA_EN": "BUSINESS BAY",
                "AREA_AR": "الخليج التجاري",
                "PROJECT_EN": "CANAL HEIGHTS ",
            },
            {
                "AREA_EN": "Business Bay",
                "AREA_AR": "الخليج التجارى",
                "PROJECT_EN": "Canal  Heights",
            },
        )
    )
    assert set(out["community"]) == {"Business Bay"}
    assert set(out["project"]) == {"Canal Heights"}


def test_report_steps_chain_and_render() -> None:
    _, report = run(dld_rows({}, {"GROUP_EN": "Mortgage"}, {"TRANS_VALUE": "1"}))
    for prev, nxt in pairwise(report.steps):
        assert prev.rows_after == nxt.rows_before
    md = report.to_markdown()
    assert md.startswith("| Step |") and "Sales only" in md


def test_synthetic_fixture_cleans_to_the_expected_rows(raw_fixture: pd.DataFrame) -> None:
    out, report = clean.clean(schema.to_canonical(raw_fixture), RULES)
    # 4 months x 140 generated sales, plus the planted penthouse and "NA"-rooms rows.
    assert len(out) == 562
    assert report.steps[0].rows_before == 613
    assert set(out["rooms"]) <= set(clean.ROOM_LEVELS)
    assert out["transaction_date"].is_monotonic_increasing
    assert "Dubai Maritime City" in set(out["community"])
    assert "Madinat Dubai Almelaheyah" not in set(out["community"])
