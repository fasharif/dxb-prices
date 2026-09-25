from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from dxb_prices import drift, report


def test_parse_handles_p_values_and_distances() -> None:
    snapshot = {
        "metrics": [
            {
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": 1.0, "share": 0.5},
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "a",
                    "method": "K-S p_value",
                    "threshold": 0.05,
                },
                "value": 0.01,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "b",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.02,
            },
        ]
    }
    out = drift._parse(snapshot)
    assert out["drifted_share"] == 0.5
    assert [(c["column"], c["drifted"]) for c in out["columns"]] == [("a", True), ("b", False)]


def test_shifted_prices_are_flagged(clean_fixture: pd.DataFrame) -> None:
    reference = clean_fixture[clean_fixture["month"] < "2026-04"]
    current = clean_fixture[clean_fixture["month"] == "2026-04"].copy()
    current["price_aed"] = current["price_aed"] * 1.6
    summary = drift.run(reference, current, html_path=None)
    by_col = {c["column"]: c for c in summary["columns"]}
    assert by_col["log_price_per_sqm"]["drifted"] is True
    assert summary["current_rows"] == len(current)


def test_empty_inputs_are_rejected(clean_fixture: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        drift.run(clean_fixture.iloc[0:0], clean_fixture, None)


def scores(mdape: float, within: float, mae: float) -> dict[str, float]:
    return {"rows": 10, "mdape": mdape, "within_10pct": within, "mae_aed": mae}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (scores(0.05, 0.8, 1e5), "beats the baseline on all three"),
        (scores(0.20, 0.3, 9e5), "does not beat the baseline on any"),
        (scores(0.05, 0.8, 9e5), "but not on mean absolute error"),
    ],
)
def test_verdict_is_plain_and_honest(model: dict[str, float], expected: str) -> None:
    text = report.verdict({"model": model, "baseline": scores(0.10, 0.5, 2e5)})
    assert expected in text


def results() -> dict[str, Any]:
    s = scores(0.05, 0.8, 1e5)
    return {
        "command": "dxb-prices train",
        "environment": {
            "python": "3.12.14",
            "platform": "Linux",
            "lightgbm": "4.7.0",
            "pandas": "3.0.6",
        },
        "data": {
            "period_start": "2026-01-01",
            "period_end": "2026-08-31",
            "downloaded_on": "2026-09-26",
            "raw_rows": 1000,
            "files": [{}] * 8,
        },
        "split": {
            "train_months": ["2026-01"],
            "valid_months": ["2026-02"],
            "test_months": ["2026-03"],
            "rows": {"train": 5, "train_after_trim": 5, "valid": 3, "final_fit": 8, "test": 2},
        },
        "chosen": {
            "params": {"objective": "huber", "num_leaves": 31, "min_data_in_leaf": 20},
            "num_boost_round": 100,
        },
        "search": [{}],
        "validation": {"model": s, "baseline": scores(0.1, 0.5, 2e5)},
        "test": {"model": s, "baseline": scores(0.1, 0.5, 2e5)},
        "interval": {"nominal": 0.8, "test_coverage": 0.79, "q10": -0.1, "q90": 0.1},
        "segments": [
            {"dimension": d, "segment": "x", "estimator": e, **s}
            for d in ("registration", "price band", "community data", "rooms")
            for e in ("model", "baseline")
        ],
    }


def test_markdown_report_contains_every_section() -> None:
    md = report.metrics_markdown(results())
    for heading in (
        "## Headline",
        "## Split",
        "## Error analysis",
        "### By price band",
        "## Estimate range",
    ):
        assert heading in md
    assert "downloaded 2026-09-26" in md
    assert "79.0%" in md


def test_readme_block_is_replaced_in_place() -> None:
    readme = f"# Title\n\n{report.README_START}\nold\n{report.README_END}\n\nMore text\n"
    updated = report.update_readme(readme, results())
    assert "old" not in updated
    assert updated.startswith("# Title") and updated.endswith("More text\n")
    assert "| Test (2026-03) | LightGBM | 10 | 5.0% | 80.0% | 100,000 |" in updated
    with pytest.raises(ValueError, match="results:start"):
        report.update_readme("# no block", results())


def test_drift_frame_encodes_categories_as_text(clean_fixture: pd.DataFrame) -> None:
    frame = drift.drift_frame(clean_fixture.head(5))
    assert frame["is_off_plan"].isin(["True", "False"]).all()
    np.testing.assert_allclose(
        frame["log_price_per_sqm"],
        np.log(clean_fixture.head(5)["price_aed"] / clean_fixture.head(5)["area_sqm"]),
    )
