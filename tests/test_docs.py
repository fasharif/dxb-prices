"""The README and the model card must agree with the published run.

`dxb-prices render-docs` writes the results blocks from reports/metrics.json;
the prose around them is written by hand. After a retrain these tests fail
until the blocks are re-rendered and every hand-written statement below still
holds, so the text is checked rather than trusted.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

from dxb_prices import report

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CARD = ROOT / "docs" / "model-card.md"


@cache
def results() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / "reports" / "metrics.json").read_text("utf-8"))
    return data


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def segments(dimension: str, estimator: str = "model") -> dict[str, dict[str, Any]]:
    return {
        s["segment"]: s
        for s in results()["segments"]
        if s["dimension"] == dimension and s["estimator"] == estimator
    }


def mdape(name: str) -> float:
    return float(results()["test"][name]["mdape"])


def test_generated_blocks_match_the_published_metrics() -> None:
    assert report.update_readme(text(README), results()) == text(README)
    assert report.update_model_card(text(CARD), results()) == text(CARD)
    assert report.metrics_markdown(results()) == text(ROOT / "reports" / "metrics.md")


def test_the_community_level_model_trails_the_full_model_and_beats_both_baselines() -> None:
    """README and model card: "beats both baselines but is clearly behind", "range is wider"."""
    test = results()["test"]
    no_project = test["model_no_project"]
    for baseline in ("baseline", "project_baseline"):
        assert no_project["mdape"] < test[baseline]["mdape"]
        assert no_project["within_10pct"] > test[baseline]["within_10pct"]
        assert no_project["mae_aed"] < test[baseline]["mae_aed"]
    assert mdape("model") < mdape("model_no_project")

    def width(variant: str) -> float:
        q = results()["interval"]["quantiles"][variant]
        return math.exp(q["q90"]) - math.exp(q["q10"])

    assert width("community") > width("full")


def test_ready_units_are_harder_than_off_plan() -> None:
    """Model card: "the median error is more than twice as large and fewer prices fall inside"."""
    reg = segments("registration")
    assert reg["ready"]["mdape"] > 2 * reg["off-plan"]["mdape"]
    assert reg["ready"]["range_coverage"] < reg["off-plan"]["range_coverage"]


def test_the_top_price_band_is_the_hardest_by_estimate_and_by_recorded_price() -> None:
    """README and model card: the most expensive homes have the largest errors and the
    lowest share of prices inside the 80% range, whichever way they are banded."""
    for dimension in ("price band", "estimated price band"):
        bands = list(segments(dimension).values())
        top = bands[-1]
        assert top["segment"] == "5.0M AED and over"
        assert top["mdape"] == max(b["mdape"] for b in bands)
        assert top["range_coverage"] == min(b["range_coverage"] for b in bands)
        assert top["range_coverage"] < results()["interval"]["nominal"]


def test_penthouses_are_few_and_the_project_median_did_better() -> None:
    """Model card: few penthouse sales, and the project median beat the model on them."""
    rooms = segments("rooms")
    project = segments("rooms", "project_baseline")
    assert rooms["penthouse"]["rows"] < 20
    assert project["penthouse"]["mdape"] < rooms["penthouse"]["mdape"]


def test_knowing_the_building_is_most_of_the_gain() -> None:
    """README findings: the project median closes most of the gap to the model."""
    gap = mdape("baseline") - mdape("model")
    assert mdape("baseline") - mdape("project_baseline") > gap / 2


def test_the_model_wins_only_where_it_knows_the_building() -> None:
    """README findings and model card: far below the baselines for buildings with a level
    of their own; the project or community median did better for rare and new buildings."""
    model, project, community = (
        segments("project data", name) for name in ("model", "project_baseline", "baseline")
    )
    own, grouped, new = (
        next(s for s in model if s.startswith(prefix)) for prefix in ("own", "grouped", "new")
    )
    assert own == "own level (40+ training sales)"
    # "Far below": at least 30% lower than the better baseline.
    assert model[own]["mdape"] < 0.7 * min(project[own]["mdape"], community[own]["mdape"])
    assert project[grouped]["mdape"] < model[grouped]["mdape"]
    assert community[new]["mdape"] < model[new]["mdape"]


def test_cold_start_statements() -> None:
    """README findings and model card on thin and new communities."""
    levels = {lv["kept_rows"]: lv["scores"] for lv in results()["cold_start"]}
    thin, new = levels[10], levels[0]
    # With 10 sales per community the model did worse than the median of those sales,
    # and far worse than with all the data...
    assert thin["baseline"]["mdape"] < thin["model"]["mdape"]
    assert thin["model"]["mdape"] > 2 * mdape("model")
    # ...and with none it would do worse still.
    assert new["model"]["mdape"] > thin["model"]["mdape"]
    # The test month's own "thin" segment is mostly one project.
    thin_segment = next(s for s in segments("community data").values() if "thin" in s["segment"])
    assert thin_segment["largest_project_share"] > 0.5


def test_the_drift_statement_names_the_flagged_columns() -> None:
    """Model card, Monitoring: size, price per square metre and community mix drifted."""
    drifted = {c["column"] for c in results()["drift"]["columns"] if c["drifted"]}
    assert drifted == {"area_sqm", "log_price_per_sqm", "community"}
