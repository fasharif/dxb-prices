from __future__ import annotations

import pandas as pd
import pytest

from dxb_prices import metrics


def test_mdape_within_and_mae_on_known_values() -> None:
    actual = [100.0, 200.0, 300.0, 400.0]
    predicted = [110.0, 180.0, 300.0, 520.0]  # errors: 10%, 10%, 0%, 30%
    assert metrics.mdape(actual, predicted) == pytest.approx(0.10)
    assert metrics.within(actual, predicted, 0.10) == pytest.approx(0.75)
    assert metrics.mae(actual, predicted) == pytest.approx((10 + 20 + 0 + 120) / 4)
    s = metrics.score(actual, predicted)
    assert s.rows == 4
    assert s.mdape == pytest.approx(0.10)
    assert s.within_10pct == pytest.approx(0.75)


def test_within_counts_the_boundary() -> None:
    assert metrics.within([1_000_000.0], [1_100_000.0], 0.10) == 1.0
    assert metrics.within([1_000_000.0], [1_100_001.0], 0.10) == 0.0


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        metrics.mdape([0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        metrics.mdape([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        metrics.score([], [])


def test_price_bands() -> None:
    bands = (0.0, 1e6, 2e6, float("inf"))
    labels = metrics.price_band_labels(pd.Series([999_999.0, 1e6, 5e6]), bands)
    assert labels.tolist() == ["under 1.0M AED", "1.0M-2.0M AED", "2.0M AED and over"]


def test_segment_table() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["a", "a", "b"],
            "price_aed": [100.0, 100.0, 100.0],
            "m": [105.0, 130.0, 100.0],
            "b": [100.0, 100.0, 150.0],
        }
    )
    table = metrics.segment_table(frame, "segment", {"model": "m", "baseline": "b"})
    row = table[(table.segment == "a") & (table.estimator == "model")].iloc[0]
    assert row["rows"] == 2 and row["within_10pct"] == 0.5
    assert set(table["estimator"]) == {"model", "baseline"}
