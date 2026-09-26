from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxb_prices import explain, features
from dxb_prices.features import COMMUNITY, FULL, VARIANT_FEATURES
from dxb_prices.model import PriceModel


@pytest.fixture(scope="module")
def rows(clean_fixture: pd.DataFrame) -> pd.DataFrame:
    return clean_fixture[clean_fixture["month"] == "2026-04"].head(40)


def test_predictions_are_positive_prices(tiny_model: PriceModel, rows: pd.DataFrame) -> None:
    pred = tiny_model.predict(rows)
    assert pred.shape == (len(rows),)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
    ratio = pred / rows["price_aed"].to_numpy()
    assert 0.5 < float(np.median(ratio)) < 2.0


def test_save_and_load_give_identical_predictions(
    tiny_model: PriceModel, rows: pd.DataFrame, tmp_path: Path
) -> None:
    tiny_model.save(tmp_path)
    again = PriceModel.load(tmp_path)
    for variant in (FULL, COMMUNITY):
        np.testing.assert_array_equal(
            again.predict(rows, variant), tiny_model.predict(rows, variant)
        )
    assert again.spec == tiny_model.spec


def test_a_model_needs_both_variants(tiny_model: PriceModel) -> None:
    with pytest.raises(ValueError, match="community"):
        PriceModel({FULL: tiny_model.boosters[FULL]}, tiny_model.spec, {})


def test_the_community_model_ignores_project_and_location_labels(
    tiny_model: PriceModel, rows: pd.DataFrame
) -> None:
    changed = rows.assign(project="Somewhere Else", nearest_metro=None, nearest_mall="X")
    np.testing.assert_array_equal(
        tiny_model.predict(changed, COMMUNITY), tiny_model.predict(rows, COMMUNITY)
    )
    assert not np.array_equal(tiny_model.predict(changed, FULL), tiny_model.predict(rows, FULL))


@pytest.mark.parametrize("variant", [FULL, COMMUNITY])
def test_contributions_add_up_to_the_prediction(
    tiny_model: PriceModel, rows: pd.DataFrame, variant: str
) -> None:
    contrib = tiny_model.contributions(rows, variant)
    assert list(contrib.columns) == [*VARIANT_FEATURES[variant], "bias"]
    np.testing.assert_allclose(
        contrib.sum(axis=1).to_numpy(),
        tiny_model.predict_log_pps(rows, variant),
        rtol=0,
        atol=1e-9,
    )


def test_top_factors_are_the_largest_contributions(
    tiny_model: PriceModel, rows: pd.DataFrame
) -> None:
    explanations = tiny_model.explain(rows.head(3), top_k=4)
    contrib = tiny_model.contributions(rows.head(3))
    for idx, explanation in zip(rows.head(3).index, explanations, strict=True):
        factors = explanation.factors
        assert len(factors) == 4
        sizes = [abs(f.contribution) for f in factors]
        assert sizes == sorted(sizes, reverse=True)
        expected_top = contrib.loc[idx, list(features.FEATURES)].abs().max()
        assert sizes[0] == pytest.approx(expected_top)
        for f in factors:
            assert f.effect_pct == pytest.approx((math.exp(f.contribution) - 1) * 100, abs=0.051)


@pytest.mark.parametrize("variant", [FULL, COMMUNITY])
def test_factors_and_the_remainder_reconcile_with_the_estimate(
    tiny_model: PriceModel, rows: pd.DataFrame, variant: str
) -> None:
    head = rows.head(3)
    bias = tiny_model.contributions(head, variant)["bias"].to_numpy()
    for i, explanation in enumerate(tiny_model.explain(head, top_k=5, variant=variant)):
        total = bias[i] + sum(f.contribution for f in explanation.factors)
        total += explanation.other_contribution
        assert total == pytest.approx(tiny_model.predict_log_pps(head, variant)[i], abs=1e-9)


@pytest.mark.parametrize("variant", [FULL, COMMUNITY])
def test_shap_library_agrees_with_lightgbm(
    tiny_model: PriceModel, rows: pd.DataFrame, variant: str
) -> None:
    values, gap = explain.shap_values(tiny_model, rows, variant)
    assert values.shape == (len(rows), len(VARIANT_FEATURES[variant]))
    assert gap < 1e-6


@pytest.mark.parametrize("variant", [FULL, COMMUNITY])
def test_interval_brackets_the_estimate(
    tiny_model: PriceModel, rows: pd.DataFrame, variant: str
) -> None:
    low, high = tiny_model.interval(rows, variant)
    pred = tiny_model.predict(rows, variant)
    assert np.all(low < pred) and np.all(pred < high)


def test_loading_a_missing_model_explains_how_to_train(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dxb-prices train"):
        PriceModel.load(tmp_path)


def test_importance_plot_is_written(
    tiny_model: PriceModel, rows: pd.DataFrame, tmp_path: Path
) -> None:
    table, _ = explain.global_importance(tiny_model, rows)
    assert table["share"].sum() == pytest.approx(1.0)
    assert list(table["mean_abs_shap"]) == sorted(table["mean_abs_shap"], reverse=True)
    out = tmp_path / "imp.png"
    explain.plot_importance(table, out, "test")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
