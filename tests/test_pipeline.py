"""End-to-end run on the synthetic fixture, including MLflow, SHAP and the drift report."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from dxb_prices import fixture, pipeline


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, clean_fixture: pd.DataFrame) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("pipeline")
    outcome = pipeline.train_and_evaluate(
        clean_fixture,
        fixture.SETTINGS,
        model_dir=root / "model",
        reports_dir=root / "reports",
        search=False,
        run_backtest=True,
        run_cold_start=True,
        data_info={
            "period_start": "2026-01-01",
            "period_end": "2026-04-30",
            "downloaded_on": "2026-05-10",
            "files": [],
            "raw_rows": 610,
        },
        track=True,
        tracking_uri=(root / "mlruns").as_uri(),
        drift_html=root / "reports" / "drift" / "drift.html",
    )
    return {"root": root, "results": outcome.results}


def test_split_holds_out_the_newest_month(run: dict[str, Any]) -> None:
    split = run["results"]["split"]
    assert split["test_months"] == ["2026-04"]
    assert split["valid_months"] == ["2026-03"]
    assert split["rows"]["final_fit"] <= split["rows"]["train"] + split["rows"]["valid"]


def test_model_learns_more_than_the_community_median(run: dict[str, Any]) -> None:
    # The synthetic prices carry project, off-plan and size effects the baseline ignores.
    test = run["results"]["test"]
    assert test["model"]["mdape"] < test["baseline"]["mdape"]
    # Knowing the project helps. (Whether the community-level model beats the community
    # median is for the real data to show; this fixture has about 20 sales per community.)
    assert test["model"]["mdape"] < test["model_no_project"]["mdape"]


def test_every_estimator_is_scored_on_the_same_test_rows(run: dict[str, Any]) -> None:
    results = run["results"]
    test, rows = results["test"], results["split"]["rows"]["test"]
    assert set(test) == set(pipeline.ESTIMATORS)
    assert {s["rows"] for s in test.values()} == {rows}
    served = results["served"]
    assert 0 < served["test_rows_project_known"] < served["test_rows"] == rows
    assert set(results["interval"]["test_coverage"]) == {"model", "model_no_project"}
    assert set(results["interval"]["quantiles"]) == {"full", "community"}


def test_served_scores_match_the_api_path(run: dict[str, Any], clean_fixture: pd.DataFrame) -> None:
    from dxb_prices import metrics, serving
    from dxb_prices.model import PriceModel

    model = PriceModel.load(run["root"] / "model")
    test = clean_fixture[clean_fixture["month"] == "2026-04"]
    for with_project, name in ((True, "model"), (False, "model_no_project")):
        rows = serving.model_rows(
            serving.requests_from_sales(test, with_project=with_project), model.spec
        )
        estimate = serving.estimate(model, rows)["estimate"]
        expected = metrics.score(test["price_aed"], estimate).mdape
        assert run["results"]["test"][name]["mdape"] == pytest.approx(expected)


def test_backtest_tests_each_month_once_with_earlier_data_only(run: dict[str, Any]) -> None:
    folds = run["results"]["backtest"]
    # Four fixture months: two to train, one to stop early, so only April can be tested.
    assert [f["test_month"] for f in folds] == ["2026-04"]
    fold = folds[0]
    assert fold["train_months"] == ["2026-01", "2026-02"] and fold["valid_months"] == ["2026-03"]
    assert set(fold["scores"]) == set(pipeline.ESTIMATORS)


def test_sales_the_api_would_refuse_are_counted_and_still_scored(run: dict[str, Any]) -> None:
    results = run["results"]
    refused = results["served"]["test_rows_the_api_would_refuse"]
    # Al Jaddaf only appears in the test month of the fixture: the API answers 404 there.
    assert refused["community without training sales (404)"] > 0
    assert results["test"]["model"]["rows"] == results["split"]["rows"]["test"]


def test_cutting_communities_keeps_a_random_few_of_their_rows(clean_fixture: pd.DataFrame) -> None:
    cut = pipeline.cut_communities(clean_fixture, {"Business Bay"}, 4, seed=1)
    assert (cut["community"] == "Business Bay").sum() == 4
    others = clean_fixture[clean_fixture["community"] != "Business Bay"]
    pd.testing.assert_frame_equal(cut[cut["community"] != "Business Bay"], others)
    pd.testing.assert_frame_equal(
        cut, pipeline.cut_communities(clean_fixture, {"Business Bay"}, 4, seed=1)
    )
    gone = pipeline.cut_communities(clean_fixture, {"Business Bay"}, 0, seed=1)
    assert "Business Bay" not in set(gone["community"])


def test_cold_start_scores_every_test_sale_at_each_level(run: dict[str, Any]) -> None:
    results = run["results"]
    levels = {lv["kept_rows"]: lv for lv in results["cold_start"]}
    assert set(levels) == {0, 10}
    for level in levels.values():
        assert level["rows"] == results["split"]["rows"]["test"]
        assert set(level["scores"]) == set(pipeline.SEGMENTED)
        assert set(level["range_coverage"]) == set(pipeline.RANGED)
    new = levels[0]
    # A community without training sales has no known project, and both baselines fall
    # back to the median of all training sales.
    assert new["rows_full_model"] == 0
    assert new["scores"]["model"] == new["scores"]["model_no_project"]
    assert new["scores"]["project_baseline"] == new["scores"]["baseline"]
    assert levels[10]["rows_full_model"] > 0


def test_segments_cover_the_required_breakdowns(run: dict[str, Any]) -> None:
    segments = run["results"]["segments"]
    dims = {s["dimension"] for s in segments}
    assert {
        "registration",
        "price band",
        "estimated price band",
        "community data",
        "project data",
        "rooms",
    } == dims
    projects = {s["segment"] for s in segments if s["dimension"] == "project data"}
    assert {"new (no training sales in its community)", "not recorded in the sale"} <= projects
    assert all("median_price_aed" not in s for s in segments)
    community = {s["segment"] for s in segments if s["dimension"] == "community data"}
    # Al Jaddaf only appears in the test month of the fixture.
    assert "new (no training sales)" in community
    new = next(
        s
        for s in segments
        if s["segment"] == "new (no training sales)" and s["estimator"] == "model"
    )
    assert new["communities"] == 1 and new["largest_project_share"] > 0.5
    reg = {s["segment"] for s in segments if s["dimension"] == "registration"}
    assert reg == {"off-plan", "ready"}


def test_reports_are_written(run: dict[str, Any]) -> None:
    reports = run["root"] / "reports"
    saved = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert saved["test"] == run["results"]["test"]
    md = (reports / "metrics.md").read_text(encoding="utf-8")
    for heading in (
        "## Headline",
        "## Rolling-origin backtest",
        "## Cold start",
        "## Drift",
        "## What drives",
    ):
        assert heading in md
    assert (reports / "shap_importance.png").exists()
    assert (reports / "drift" / "drift.html").stat().st_size > 10_000
    assert run["results"]["shap"]["max_abs_gap_shap_vs_lightgbm"] < 1e-6


def test_drift_summary_names_each_column(run: dict[str, Any]) -> None:
    drift = run["results"]["drift"]
    columns = {c["column"] for c in drift["columns"]}
    assert {"area_sqm", "log_price_per_sqm", "community", "rooms"} <= columns
    assert drift["current_month"] == "2026-04"


def test_mlflow_run_is_recorded(run: dict[str, Any]) -> None:
    import mlflow

    from dxb_prices import tracking

    mlflow.set_tracking_uri(tracking.resolve_uri((run["root"] / "mlruns").as_uri()))
    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT])
    parent = runs[runs["tags.mlflow.runName"] == "train-2026-04"]
    assert len(parent) == 1
    row = parent.iloc[0]
    assert row["metrics.test_model_mdape"] == pytest.approx(
        run["results"]["test"]["model"]["mdape"]
    )
    assert "metrics.test_baseline_mdape" in parent.columns
    assert "metrics.test_model_no_project_mdape" in parent.columns
    assert "metrics.backtest_2026-04_model_mdape" in parent.columns
    assert "metrics.cold_start_0_model_mdape" in parent.columns
    client = mlflow.tracking.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(row["run_id"])}
    assert {"model", "reports"} <= artifacts


def test_model_metadata_is_complete(run: dict[str, Any]) -> None:
    meta = json.loads((run["root"] / "model" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["trained_on_months"] == ["2026-01", "2026-02", "2026-03"]
    assert meta["evaluated_on_months"] == ["2026-04"]
    assert meta["data_period_end"] == "2026-04-30"
    assert set(meta["residual_quantiles"]) == {"full", "community"}
    assert set(meta["residual_quantiles"]["full"]) == {"q10", "q90"}
    assert set(meta["base_per_sqm"]) == {"full", "community"}
