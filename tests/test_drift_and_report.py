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
    return {
        "rows": 10,
        "mdape": mdape,
        "within_10pct": within,
        "mae_aed": mae,
        "median_error": -0.02,
    }


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            scores(0.05, 0.8, 1e5),
            "beats both the community-median and the project-median baseline on all three",
        ),
        (scores(0.20, 0.3, 9e5), "does not beat the community-median baseline on any"),
        (
            scores(0.05, 0.8, 9e5),
            "beats the community-median baseline on MdAPE and share within 10% but not on MAE",
        ),
    ],
)
def test_verdict_is_plain_and_honest(model: dict[str, float], expected: str) -> None:
    base = scores(0.10, 0.5, 2e5)
    text = report.verdict(
        {"model": model, "project_baseline": base, "baseline": base, "model_no_project": base}
    )
    assert text.startswith("With the project, LightGBM " + expected)
    assert "Without the project, the community-level model does not beat" in text
    assert "the project median alone moves MdAPE from 10.0% to 10.0%" in text


@pytest.mark.parametrize(
    ("project_mdape", "expected"),
    [
        (0.07, "Most of the gain over the community median comes from knowing the building"),
        (0.09, "Knowing the building gives less than half of the gain"),
        (0.11, "On MdAPE, the project median alone moves MdAPE from 10.0% to 11.0%"),
    ],
)
def test_the_building_sentence_follows_the_numbers(project_mdape: float, expected: str) -> None:
    test = {
        "baseline": scores(0.10, 0.5, 2e5),
        "project_baseline": scores(project_mdape, 0.5, 2e5),
        "model": scores(0.05, 0.8, 1e5),
    }
    assert report.building_share(test).startswith(expected)


ESTIMATORS = ("model", "model_no_project", "project_baseline", "baseline", "model_recorded")


def results() -> dict[str, Any]:
    s = scores(0.05, 0.8, 1e5)
    by_estimator = dict.fromkeys(ESTIMATORS, s) | {"baseline": scores(0.1, 0.5, 2e5)}
    profile = {"communities": 3, "largest_project_share": 0.4}
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
            "num_boost_round_community": 80,
        },
        "search": [{}],
        "served": {
            "test_rows": 2,
            "test_rows_project_known": 1,
            "min_project_rows_baseline": 5,
            "test_rows_the_api_would_refuse": {
                "community without training sales (404)": 1,
                "room count not recorded (422)": 0,
            },
        },
        "validation": by_estimator,
        "test": by_estimator,
        "interval": {
            "nominal": 0.8,
            "quantiles": {
                "full": {"q10": -0.1, "q90": 0.1},
                "community": {"q10": -0.2, "q90": 0.2},
            },
            "test_coverage": {"model": 0.79, "model_no_project": 0.81},
        },
        "segments": [
            {"dimension": d, "segment": "x", "estimator": e, **s, **profile}
            | ({"range_coverage": 0.7} if e == "model" else {})
            for d in report.SEGMENT_DIMENSIONS
            for e in ESTIMATORS[:4]
        ],
        "backtest": [
            {
                "test_month": m,
                "train_months": ["2026-01"],
                "valid_months": ["2026-02"],
                "rows": 4,
                "scores": by_estimator,
            }
            for m in ("2026-03", "2026-04")
        ],
        "cold_start": [
            {
                "kept_rows": kept,
                "groups": 5,
                "communities": 20,
                "rows": 2,
                "rows_full_model": 1 if kept else 0,
                "scores": dict.fromkeys(ESTIMATORS[:4], scores(mdape, 0.6, 2e5)),
                "range_coverage": {"model": 0.7, "model_no_project": 0.72},
            }
            for kept, mdape in ((0, 0.3), (10, 0.08))
        ],
    }


def test_markdown_report_contains_every_section() -> None:
    md = report.metrics_markdown(results())
    for heading in (
        "## Headline",
        "## Rolling-origin backtest",
        "## Validation month",
        "## Split",
        "## Error analysis",
        "### By price band",
        "### By estimated price band",
        "### By project data",
        "## Cold start",
        "## Estimate range",
    ):
        assert heading in md
    assert "Median price" not in md
    assert "| 0: a new community (the API refuses these) | 10 | 30.0% |" in md
    assert "The API would refuse 1 test sale (community without training sales (404): 1)" in md
    assert "downloaded 2026-09-26" in md
    assert "contained 79.0% of test prices with the project and 81.0% without" in md
    assert "| LightGBM, no project (community-level model) | 10 | 5.0% |" in md
    assert "Largest single project" in md
    assert "(1 of 2), get the community-level model" in md


def test_readme_blocks_are_replaced_in_place() -> None:
    readme = (
        f"# Title\n\nPitch.\n{report.HEADLINE_START}\nOLD\n{report.HEADLINE_END}\n\n"
        f"{report.README_START}\nSTALE\n{report.README_END}\n\nMore text\n"
    )
    updated = report.update_readme(readme, results())
    assert "STALE" not in updated and "OLD" not in updated
    assert updated.startswith("# Title") and updated.endswith("More text\n")
    assert (
        "On the 2 sales of March 2026, which the models had not seen, the median error was 5.0% "
        "when the building is known and 5.0% when it is not, against 10.0%"
    ) in updated
    assert "| LightGBM, project given | 10 | 5.0% | 80.0% | 100,000 | -2.0% | 79.0% |" in updated
    assert "<summary>What each row means</summary>" in updated
    assert "Rolling-origin backtest over 2 test months (2026-03 to 2026-04" in updated
    assert "each community in turn cut to 10 training sales" in updated
    assert "had it answered, MdAPE would have been 30.0%" in updated
    assert "cannot be downloaded again with this tool after 31 December 2026" in updated
    with pytest.raises(ValueError, match="headline:start"):
        report.update_readme("# no block", results())


def test_drift_frame_encodes_categories_as_text(clean_fixture: pd.DataFrame) -> None:
    frame = drift.drift_frame(clean_fixture.head(5))
    assert frame["is_off_plan"].isin(["True", "False"]).all()
    np.testing.assert_allclose(
        frame["log_price_per_sqm"],
        np.log(clean_fixture.head(5)["price_aed"] / clean_fixture.head(5)["area_sqm"]),
    )
