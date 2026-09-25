from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dxb_prices import split
from dxb_prices.config import SplitRules, TrainingTrim


def months(n: int) -> list[str]:
    return [f"2026-{m:02d}" for m in range(1, n + 1)]


def test_newest_months_are_held_out() -> None:
    plan = split.plan(months(8) * 2, SplitRules())
    assert plan.train_months == tuple(months(6))
    assert plan.valid_months == ("2026-07",)
    assert plan.test_months == ("2026-08",)
    assert plan.describe() == "train 2026-01 to 2026-06, validate 2026-07, test 2026-08"


def test_too_few_months_is_an_explicit_error() -> None:
    with pytest.raises(split.InsufficientDataError, match="need at least 4 months"):
        split.plan(months(3), SplitRules())


def test_apply_gives_disjoint_ordered_periods(clean_fixture: pd.DataFrame) -> None:
    plan = split.plan(clean_fixture["month"], SplitRules())
    train, valid, test = split.apply(clean_fixture, plan)
    assert len(train) + len(valid) + len(test) == len(clean_fixture)
    assert train["transaction_date"].max() < valid["transaction_date"].min()
    assert valid["transaction_date"].max() < test["transaction_date"].min()


def test_trim_removes_only_extreme_training_rows() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    train = pd.DataFrame(
        {
            "community": ["A"] * n,
            "area_sqm": 100.0,
            "price_aed": 100.0 * 15_000 * rng.lognormal(0, 0.1, n),
        }
    )
    train.loc[0, "price_aed"] = 100.0 * 150_000  # a tenfold outlier
    trimmed = split.trim_training(train, TrainingTrim(0.005, 0.995))
    assert 0 not in trimmed.index
    assert 985 <= len(trimmed) <= 992


def test_small_communities_use_global_quantiles() -> None:
    big = pd.DataFrame(
        {"community": "A", "area_sqm": 100.0, "price_aed": np.linspace(1.0e6, 2.0e6, 400)}
    )
    small = pd.DataFrame({"community": "B", "area_sqm": 100.0, "price_aed": [1.5e6] * 5})
    trimmed = split.trim_training(pd.concat([big, small], ignore_index=True), TrainingTrim())
    assert (trimmed["community"] == "B").sum() == 5
