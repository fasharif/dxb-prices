from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dxb_prices.baseline import CommunityMedianBaseline, ProjectMedianBaseline


def sales(community: str, pps: list[float], area: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"community": community, "area_sqm": area, "price_aed": [p * area for p in pps]}
    )


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.concat(
        [sales("A", [10_000, 12_000, 30_000]), sales("B", [20_000, 22_000])], ignore_index=True
    )


def test_uses_the_community_median_price_per_sqm(train: pd.DataFrame) -> None:
    model = CommunityMedianBaseline.fit(train)
    assert model.medians == {"A": 12_000.0, "B": 21_000.0}
    test = pd.DataFrame({"community": ["A", "B"], "area_sqm": [50.0, 80.0]})
    np.testing.assert_allclose(model.predict(test), [600_000.0, 1_680_000.0])


def test_unseen_communities_fall_back_to_the_training_median(train: pd.DataFrame) -> None:
    model = CommunityMedianBaseline.fit(train)
    test = pd.DataFrame({"community": ["C"], "area_sqm": [100.0]})
    assert model.global_median == 20_000.0
    assert model.predict(test)[0] == 2_000_000.0
    assert model.covered(test).tolist() == [False]


def test_test_prices_never_influence_the_baseline(train: pd.DataFrame) -> None:
    model = CommunityMedianBaseline.fit(train)
    test = pd.DataFrame({"community": ["A"], "area_sqm": [100.0], "price_aed": [1.0]})
    before = model.predict(test)
    test["price_aed"] = 1e12
    np.testing.assert_array_equal(model.predict(test), before)


def test_round_trip(train: pd.DataFrame) -> None:
    model = CommunityMedianBaseline.fit(train)
    assert CommunityMedianBaseline.from_dict(model.to_dict()) == model


def test_project_median_needs_enough_sales_and_falls_back_to_the_community() -> None:
    train = pd.concat(
        [
            sales("A", [10_000, 11_000, 12_000]).assign(project="Tower 1"),
            sales("A", [30_000, 31_000]).assign(project="Tower 2"),
            sales("B", [20_000, 22_000, 24_000]).assign(project="Tower 1"),
        ],
        ignore_index=True,
    )
    model = ProjectMedianBaseline.fit(train, min_rows=3)
    # Tower 2 has two sales: below the minimum, so community A's median (12,000) applies.
    assert model.medians == {"A": {"Tower 1": 11_000.0}, "B": {"Tower 1": 22_000.0}}
    test = pd.DataFrame(
        {
            "community": ["A", "A", "B", "A", "C"],
            "project": ["Tower 1", "Tower 2", "Tower 1", None, "Tower 1"],
            "area_sqm": [100.0] * 5,
        }
    )
    # The same project name in another community is a different building, and a community
    # without training sales falls back to the median of all of them (21,000).
    np.testing.assert_allclose(
        model.per_sqm(test), [11_000.0, 12_000.0, 22_000.0, 12_000.0, 21_000.0]
    )
    np.testing.assert_allclose(model.predict(test)[:1], [1_100_000.0])


def test_refuses_empty_training_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        CommunityMedianBaseline.fit(sales("A", []))
