"""Model rows built from what a caller supplies (shared by the API and the evaluation)."""

from __future__ import annotations

import pandas as pd
import pytest

from dxb_prices import features, serving
from dxb_prices.features import COMMUNITY, FULL
from dxb_prices.model import PriceModel
from tests.test_features import RULES, training_frame


def request(**changes: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "community": "Business Bay",
        "project": "Canal Heights",
        "rooms": "1",
        "sub_type": "Flat",
        "area_sqm": 75.0,
        "is_off_plan": False,
        "is_freehold": None,
        "month": "2026-03",
    }
    return pd.DataFrame([base | changes])


@pytest.fixture
def spec() -> features.FeatureSpec:
    train = training_frame()
    # Give Marina Crest (Marsa Dubai) its own landmark so project and community values differ.
    train.loc[train["project"] == "Marina Crest", "nearest_landmark"] = "Marina Walk"
    return features.fit(train, RULES)


def test_a_known_project_uses_the_full_model_and_its_own_location_labels(
    spec: features.FeatureSpec,
) -> None:
    rows = serving.model_rows(request(community="Marsa Dubai", project="Marina Crest"), spec)
    assert rows.loc[0, "variant"] == FULL
    assert rows.loc[0, "nearest_landmark"] == "Marina Walk"
    assert bool(rows.loc[0, "is_freehold"]) is True


def test_a_missing_or_unknown_project_uses_the_community_model(
    spec: features.FeatureSpec,
) -> None:
    for project in (None, "Nowhere Towers"):
        rows = serving.model_rows(request(project=project), spec)
        assert rows.loc[0, "variant"] == COMMUNITY
        assert pd.isna(rows.loc[0, "project"])
        assert rows.loc[0, "nearest_mall"] == "Dubai Mall"  # the community's usual value


def test_a_project_recorded_only_in_another_community_uses_the_community_model(
    spec: features.FeatureSpec,
) -> None:
    # Marina Crest has training sales in Marsa Dubai, none in Business Bay.
    rows = serving.model_rows(request(community="Business Bay", project="Marina Crest"), spec)
    assert rows.loc[0, "variant"] == COMMUNITY
    assert pd.isna(rows.loc[0, "project"])
    assert rows.loc[0, "nearest_landmark"] == "Burj Khalifa"  # Business Bay's, not Marina Walk


def test_sales_the_api_would_refuse_are_counted(spec: features.FeatureSpec) -> None:
    sales = pd.concat(
        [
            request(),
            request(community="Al Jaddaf"),
            request(rooms="unknown"),
            request(community="Al Jaddaf", rooms="unknown"),
        ],
        ignore_index=True,
    )
    assert serving.refusals(sales, spec) == {
        "community without training sales (404)": 2,
        "room count not recorded (422)": 1,
    }


def test_the_api_accepts_the_same_room_counts_the_refusal_count_assumes() -> None:
    from typing import get_args

    from dxb_prices.api.schemas import RoomsLabel

    assert get_args(RoomsLabel) == serving.API_ROOMS


def test_a_given_freehold_flag_is_kept(spec: features.FeatureSpec) -> None:
    rows = serving.model_rows(request(is_freehold=False), spec)
    assert bool(rows.loc[0, "is_freehold"]) is False


def test_an_unknown_community_leaves_the_labels_empty(spec: features.FeatureSpec) -> None:
    rows = serving.model_rows(request(community="Al Jaddaf", project=None), spec)
    assert rows.loc[0, "variant"] == COMMUNITY
    assert pd.isna(rows.loc[0, "nearest_metro"]) and pd.isna(rows.loc[0, "is_freehold"])


def test_missing_request_columns_are_named(spec: features.FeatureSpec) -> None:
    with pytest.raises(KeyError, match="month"):
        serving.model_rows(request().drop(columns=["month"]), spec)


def test_recorded_sales_become_requests_without_labels_or_freehold(
    clean_fixture: pd.DataFrame,
) -> None:
    sales = clean_fixture.head(5)
    with_project = serving.requests_from_sales(sales, with_project=True)
    assert list(with_project.columns) == list(serving.REQUEST_COLUMNS)
    assert with_project["is_freehold"].isna().all()
    assert with_project["project"].tolist() == sales["project"].tolist()
    assert serving.requests_from_sales(sales, with_project=False)["project"].isna().all()


def test_estimates_follow_each_rows_model(
    tiny_model: PriceModel, clean_fixture: pd.DataFrame
) -> None:
    sales = clean_fixture[clean_fixture["month"] == "2026-04"]
    rows = serving.model_rows(
        serving.requests_from_sales(sales, with_project=True), tiny_model.spec
    )
    assert set(rows["variant"]) == {FULL, COMMUNITY}  # the fixture has sales without a project
    out = serving.estimate(tiny_model, rows)
    for variant in (FULL, COMMUNITY):
        part = rows[rows["variant"] == variant]
        pd.testing.assert_series_equal(
            out.loc[part.index, "estimate"],
            pd.Series(tiny_model.predict(part, variant), index=part.index, name="estimate"),
        )
    assert (out["low"] < out["estimate"]).all() and (out["estimate"] < out["high"]).all()
