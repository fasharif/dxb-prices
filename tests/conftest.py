"""Shared fixtures: the synthetic CSV, its cleaned form and a tiny trained model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from dxb_prices import schema

if TYPE_CHECKING:
    from dxb_prices.model import PriceModel

# Heavier modules are imported inside the fixtures that need them, so tests of
# the lower layers (schema, download, cleaning) do not load LightGBM.

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "transactions_synthetic.csv"


def dld_rows(*rows: dict[str, Any]) -> pd.DataFrame:
    """Build a raw frame in the DLD export layout; unspecified columns get sensible defaults."""
    defaults: dict[str, Any] = {
        "TRANSACTION_NUMBER": None,
        "INSTANCE_DATE": "2026-03-10 10:00:00",
        "GROUP_EN": "Sales",
        "PROCEDURE_EN": "Sale",
        "IS_OFFPLAN_EN": "Ready",
        "IS_FREE_HOLD_EN": "Free Hold",
        "USAGE_EN": "Residential",
        "AREA_EN": "Business Bay",
        "AREA_AR": "الخليج التجاري",
        "PROP_TYPE_EN": "Unit",
        "PROP_SB_TYPE_EN": "Flat",
        "TRANS_VALUE": "1500000",
        "PROCEDURE_AREA": "75",
        "ACTUAL_AREA": "75",
        "ROOMS_EN": "1 B/R",
        "PARKING": "1",
        "NEAREST_METRO_EN": "Business Bay Metro Station",
        "NEAREST_MALL_EN": "Dubai Mall",
        "NEAREST_LANDMARK_EN": "Burj Khalifa",
        "TOTAL_BUYER": "1",
        "TOTAL_SELLER": "1",
        "MASTER_PROJECT_EN": "",
        "MASTER_PROJECT_AR": "",
        "PROJECT_EN": "Canal Heights",
        "PROJECT_AR": "",
    }
    out = []
    for i, row in enumerate(rows):
        merged = defaults | {"TRANSACTION_NUMBER": f"102-{i + 1}-2026"} | row
        out.append({k: "" if v is None else str(v) for k, v in merged.items()})
    return pd.DataFrame(out)


@pytest.fixture(scope="session")
def raw_fixture() -> pd.DataFrame:
    return schema.read_raw_csvs([FIXTURE_CSV])


@pytest.fixture(scope="session")
def clean_fixture(raw_fixture: pd.DataFrame) -> pd.DataFrame:
    from dxb_prices import clean, fixture

    cleaned, _ = clean.clean(schema.to_canonical(raw_fixture), fixture.SETTINGS.cleaning)
    return cleaned


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory, clean_fixture: pd.DataFrame) -> Path:
    from dxb_prices import fixture, pipeline

    directory = tmp_path_factory.mktemp("tiny-model")
    pipeline.train_and_evaluate(
        clean_fixture, fixture.SETTINGS, model_dir=directory, reports_dir=None, search=False
    )
    return directory


@pytest.fixture(scope="session")
def tiny_model(tiny_model_dir: Path) -> PriceModel:
    from dxb_prices.model import PriceModel

    return PriceModel.load(tiny_model_dir)
