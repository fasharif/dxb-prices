from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxb_prices import explain, features
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
    np.testing.assert_array_equal(again.predict(rows), tiny_model.predict(rows))
    assert again.spec == tiny_model.spec


def test_contributions_add_up_to_the_prediction(tiny_model: PriceModel, rows: pd.DataFrame) -> None:
    contrib = tiny_model.contributions(rows)
    np.testing.assert_allclose(
        contrib.sum(axis=1).to_numpy(), tiny_model.predict_log_pps(rows), rtol=0, atol=1e-9
    )


def test_top_factors_are_the_largest_contributions(
    tiny_model: PriceModel, rows: pd.DataFrame
) -> None:
    factors = tiny_model.explain(rows.head(3), top_k=4)
    contrib = tiny_model.contributions(rows.head(3))
    for idx, row_factors in zip(rows.head(3).index, factors, strict=True):
        assert len(row_factors) == 4
        sizes = [abs(f.contribution) for f in row_factors]
        assert sizes == sorted(sizes, reverse=True)
        expected_top = contrib.loc[idx, list(features.FEATURES)].abs().max()
        assert sizes[0] == pytest.approx(expected_top)
        for f in row_factors:
            assert f.effect_pct == pytest.approx((math.exp(f.contribution) - 1) * 100, abs=0.051)


def test_shap_library_agrees_with_lightgbm(tiny_model: PriceModel, rows: pd.DataFrame) -> None:
    values, gap = explain.shap_values(tiny_model, rows)
    assert values.shape == (len(rows), len(features.FEATURES))
    assert gap < 1e-6


def test_interval_brackets_the_estimate(tiny_model: PriceModel, rows: pd.DataFrame) -> None:
    low, high = tiny_model.interval(rows)
    pred = tiny_model.predict(rows)
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
