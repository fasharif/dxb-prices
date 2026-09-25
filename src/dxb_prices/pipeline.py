"""End-to-end training and evaluation.

Order of work, and which rows each step may see:

1. Split by month: oldest months train, the next month validates, the newest
   month tests.
2. Model selection (a small hyper-parameter search with early stopping) uses
   the training months to fit and the validation month to score. The test
   month is not touched.
3. The chosen settings are refitted on training + validation months, and the
   baseline medians are recomputed on the same rows. Only then is the test
   month scored, once, for both the model and the baseline.
"""

from __future__ import annotations

import json
import logging
import platform
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from dxb_prices import baseline as baseline_mod
from dxb_prices import clean as clean_mod
from dxb_prices import features, metrics, report, schema, split
from dxb_prices.config import Settings
from dxb_prices.model import BASE_PARAMS, SEARCH_SPACE, PriceModel, log_price_per_sqm, train_booster

log = logging.getLogger(__name__)


@dataclass
class TrainingOutcome:
    model: PriceModel
    results: dict[str, Any]


def load_and_clean(
    raw_paths: list[Path], settings: Settings
) -> tuple[pd.DataFrame, clean_mod.CleaningReport, int]:
    raw = schema.read_raw_csvs(raw_paths)
    canonical = schema.to_canonical(raw)
    cleaned, cleaning_report = clean_mod.clean(canonical, settings.cleaning)
    return cleaned, cleaning_report, len(raw)


def _community_segment(frame: pd.DataFrame, rows: dict[str, int], thin: int) -> pd.Series:
    counts = frame["community"].astype("object").map(rows).fillna(0)
    order = [
        "new (no training sales)",
        f"thin (1-{thin - 1} training sales)",
        f"established ({thin}+ training sales)",
    ]
    labels = np.where(counts == 0, order[0], np.where(counts < thin, order[1], order[2]))
    return pd.Series(pd.Categorical(labels, categories=order), index=frame.index)


def segment_scores(
    scored: pd.DataFrame, community_rows: dict[str, int], settings: Settings
) -> list[dict[str, Any]]:
    frame = scored.copy()
    frame["registration"] = pd.Categorical(
        np.where(frame["is_off_plan"].astype("boolean").fillna(False), "off-plan", "ready"),
        categories=["off-plan", "ready"],
    )
    frame["price_band"] = metrics.price_band_labels(
        frame["price_aed"], settings.segments.price_bands
    )
    frame["community_data"] = _community_segment(
        frame, community_rows, settings.segments.thin_community_rows
    )
    frame["rooms_segment"] = pd.Categorical(frame["rooms"], categories=clean_mod.ROOM_LEVELS)
    in_range = None
    if {"range_low", "range_high"} <= set(frame.columns):
        in_range = (frame["price_aed"] >= frame["range_low"]) & (
            frame["price_aed"] <= frame["range_high"]
        )
    out: list[dict[str, Any]] = []
    for dimension, col in (
        ("registration", "registration"),
        ("price band", "price_band"),
        ("community data", "community_data"),
        ("rooms", "rooms_segment"),
    ):
        table = metrics.segment_table(
            frame, col, {"model": "pred_model", "baseline": "pred_baseline"}
        )
        coverage = (
            in_range.groupby(frame[col], observed=True).mean() if in_range is not None else None
        )
        for rec in table.to_dict(orient="records"):
            if coverage is not None and rec["estimator"] == "model":
                rec["range_coverage"] = float(coverage[rec["segment"]])
            out.append({"dimension": dimension, **{str(k): v for k, v in rec.items()}})
    return out


def _search(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    candidates: tuple[dict[str, Any], ...],
    track: bool,
) -> tuple[dict[str, Any], lgb.Booster, list[dict[str, Any]]]:
    y_va = log_price_per_sqm(valid)
    area = valid["area_sqm"].to_numpy(np.float64)
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], lgb.Booster] | None = None
    for i, params in enumerate(candidates):
        booster = train_booster(x_tr, y_tr, params, x_valid=x_va, y_valid=y_va)
        pred = np.exp(booster.predict(x_va, num_iteration=booster.best_iteration)) * area
        s = metrics.score(valid["price_aed"], pred)
        trials.append(
            {"params": params, "best_iteration": int(booster.best_iteration), **s.as_dict()}
        )
        log.info(
            "trial %d/%d %s -> validation MdAPE %.4f (%d rounds)",
            i + 1,
            len(candidates),
            params,
            s.mdape,
            booster.best_iteration,
        )
        if track:
            from dxb_prices import tracking

            with tracking.child_run(f"search-{i + 1:02d}"):
                tracking.log_params(params | {"best_iteration": booster.best_iteration})
                tracking.log_metrics(
                    {f"valid_{k}": v for k, v in s.as_dict().items() if k != "rows"}
                )
        if best is None or s.mdape < best[0]:
            best = (s.mdape, params, booster)
    assert best is not None
    return best[1], best[2], trials


@dataclass
class _Selection:
    params: dict[str, Any]
    num_boost_round: int
    trials: list[dict[str, Any]]
    residual_quantiles: dict[str, float]
    scores: dict[str, dict[str, Any]]
    rows_after_trim: int


def _select(
    train: pd.DataFrame, valid: pd.DataFrame, settings: Settings, *, search: bool, track: bool
) -> _Selection:
    """Step 1: fit on the training months, choose settings on the validation month."""
    train_fit = split.trim_training(train, settings.trim)
    spec = features.fit(train_fit, settings.features)
    x_va = features.transform(valid, spec)
    params, booster, trials = _search(
        features.transform(train_fit, spec),
        log_price_per_sqm(train_fit),
        x_va,
        valid,
        candidates=SEARCH_SPACE if search else ({},),
        track=track,
    )
    rounds = int(booster.best_iteration) or int(booster.current_iteration())
    log_pred = booster.predict(x_va, num_iteration=rounds)
    residuals = log_price_per_sqm(valid) - log_pred
    baseline = baseline_mod.CommunityMedianBaseline.fit(train_fit)
    return _Selection(
        params=params,
        num_boost_round=rounds,
        trials=trials,
        residual_quantiles={
            "q10": float(np.quantile(residuals, 0.10)),
            "q90": float(np.quantile(residuals, 0.90)),
        },
        scores={
            "model": metrics.score(
                valid["price_aed"], np.exp(log_pred) * valid["area_sqm"].to_numpy()
            ).as_dict(),
            "baseline": metrics.score(valid["price_aed"], baseline.predict(valid)).as_dict(),
        },
        rows_after_trim=len(train_fit),
    )


@dataclass
class _Evaluation:
    model: PriceModel
    baseline: baseline_mod.CommunityMedianBaseline
    fit_rows: pd.DataFrame
    scores: dict[str, dict[str, Any]]
    coverage: float
    segments: list[dict[str, Any]]


def _refit_and_test(
    trainval: pd.DataFrame, test: pd.DataFrame, settings: Settings, selection: _Selection
) -> _Evaluation:
    """Step 2: refit on training + validation months, then score the test month once."""
    fit_rows = split.trim_training(trainval, settings.trim)
    spec = features.fit(fit_rows, settings.features)
    booster = train_booster(
        features.transform(fit_rows, spec),
        log_price_per_sqm(fit_rows),
        selection.params,
        num_boost_round=selection.num_boost_round,
    )
    model = PriceModel(booster, spec, metadata={"residual_quantiles": selection.residual_quantiles})
    base = baseline_mod.CommunityMedianBaseline.fit(fit_rows)
    scored = test.copy()
    scored["pred_model"] = model.predict(test)
    scored["pred_baseline"] = base.predict(test)
    scored["range_low"], scored["range_high"] = model.interval(test)
    in_range = (scored["price_aed"] >= scored["range_low"]) & (
        scored["price_aed"] <= scored["range_high"]
    )
    return _Evaluation(
        model=model,
        baseline=base,
        fit_rows=fit_rows,
        scores={
            "model": metrics.score(scored["price_aed"], scored["pred_model"]).as_dict(),
            "baseline": metrics.score(scored["price_aed"], scored["pred_baseline"]).as_dict(),
        },
        coverage=float(in_range.mean()),
        segments=segment_scores(scored, spec.community_rows, settings),
    )


def _results(
    temporal: split.TemporalSplit,
    rows: dict[str, int],
    selection: _Selection,
    evaluation: _Evaluation,
    *,
    search: bool,
    data_info: dict[str, Any],
    cleaning_report: clean_mod.CleaningReport | None,
) -> dict[str, Any]:
    spec = evaluation.model.spec
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "dxb-prices train" + ("" if search else " --no-search"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
        },
        "data": data_info,
        "cleaning": cleaning_report.as_records() if cleaning_report else [],
        "split": {
            "train_months": list(temporal.train_months),
            "valid_months": list(temporal.valid_months),
            "test_months": list(temporal.test_months),
            "rows": rows
            | {
                "train_after_trim": selection.rows_after_trim,
                "final_fit": len(evaluation.fit_rows),
            },
            "final_fit_communities": int(evaluation.fit_rows["community"].nunique()),
            "final_fit_projects": int(evaluation.fit_rows["project"].nunique()),
            "communities_with_own_level": len(spec.levels["community"]) - 2,
            "projects_with_own_level": len(spec.levels["project"]) - 2,
        },
        "search": selection.trials,
        "chosen": {
            "params": BASE_PARAMS | selection.params,
            "num_boost_round": selection.num_boost_round,
        },
        "validation": selection.scores,
        "test": evaluation.scores,
        "interval": {
            "nominal": 0.8,
            "test_coverage": evaluation.coverage,
            **selection.residual_quantiles,
        },
        "segments": evaluation.segments,
    }


def _model_metadata(
    temporal: split.TemporalSplit,
    results: dict[str, Any],
    evaluation: _Evaluation,
    settings: Settings,
) -> dict[str, Any]:
    probe = evaluation.fit_rows.head(1)
    bias = float(evaluation.model.contributions(probe)["bias"].iloc[0])
    return {
        "model_version": f"{temporal.test_months[-1]}-{datetime.now(UTC):%Y%m%d%H%M%S}",
        "trained_at": results["generated_at"],
        "target": "natural log of price per square metre",
        "trained_on_months": list(temporal.train_months + temporal.valid_months),
        "evaluated_on_months": list(temporal.test_months),
        "data_period_end": results["data"].get("period_end"),
        "residual_quantiles": {k: results["interval"][k] for k in ("q10", "q90")},
        "base_per_sqm": float(np.exp(bias)),
        "baseline": evaluation.baseline.to_dict(),
        "test_scores": results["test"],
        "thin_community_rows": settings.segments.thin_community_rows,
        "params": results["chosen"],
    }


def _write_reports(
    results: dict[str, Any],
    model: PriceModel,
    reference: pd.DataFrame,
    test: pd.DataFrame,
    reports_dir: Path,
    *,
    drift_html: Path | None,
    shap_sample: int,
    seed: int,
) -> None:
    from dxb_prices import drift, explain

    reports_dir.mkdir(parents=True, exist_ok=True)
    test_month = results["split"]["test_months"][-1]
    importance, gap = explain.global_importance(model, test, sample=shap_sample, seed=seed)
    results["shap"] = {
        "rows": min(shap_sample, len(test)),
        "max_abs_gap_shap_vs_lightgbm": gap,
        "importance": importance[["feature", "label", "mean_abs_shap", "share"]].to_dict(
            orient="records"
        ),
    }
    explain.plot_importance(
        importance,
        reports_dir / "shap_importance.png",
        f"What moves the estimate: mean |SHAP| on the {test_month} test month",
    )
    results["drift"] = drift.run(reference, test, drift_html) | {
        "reference_months": results["split"]["train_months"] + results["split"]["valid_months"],
        "current_month": test_month,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "metrics.md").write_text(report.metrics_markdown(results), encoding="utf-8")


def _log_to_mlflow(results: dict[str, Any], model_dir: Path, reports_dir: Path | None) -> None:
    from dxb_prices import tracking

    split_info = results["split"]
    tracking.log_params(
        {
            "lgbm": results["chosen"],
            "split": {
                k: ",".join(split_info[k]) for k in ("train_months", "valid_months", "test_months")
            },
            "data": {k: v for k, v in results["data"].items() if k != "files"},
        }
    )
    flat = {
        f"{stage}_{name}_{k}": v
        for stage in ("validation", "test")
        for name, s in results[stage].items()
        for k, v in s.items()
        if k != "rows"
    }
    flat["test_interval80_coverage"] = results["interval"]["test_coverage"]
    tracking.log_metrics(flat)
    tracking.log_artifacts(model_dir, "model")
    if reports_dir is not None:
        tracking.log_artifacts(reports_dir, "reports")


def train_and_evaluate(
    cleaned: pd.DataFrame,
    settings: Settings,
    *,
    model_dir: Path,
    reports_dir: Path | None,
    search: bool = True,
    cleaning_report: clean_mod.CleaningReport | None = None,
    data_info: dict[str, Any] | None = None,
    track: bool = False,
    tracking_uri: str | None = None,
    drift_html: Path | None = None,
    shap_sample: int = 5000,
) -> TrainingOutcome:
    temporal = split.plan(cleaned["month"], settings.split)
    train, valid, test = split.apply(cleaned, temporal)
    log.info(
        "split: %s (%d / %d / %d rows)", temporal.describe(), len(train), len(valid), len(test)
    )
    with ExitStack() as stack:
        if track:
            from dxb_prices import tracking

            stack.enter_context(
                tracking.run(
                    tracking_uri,
                    run_name=f"train-{temporal.test_months[-1]}",
                    tags={"test_month": temporal.test_months[-1]},
                )
            )
        selection = _select(train, valid, settings, search=search, track=track)
        trainval = pd.concat([train, valid])
        evaluation = _refit_and_test(trainval, test, settings, selection)
        results = _results(
            temporal,
            {"train": len(train), "valid": len(valid), "test": len(test)},
            selection,
            evaluation,
            search=search,
            data_info=data_info or {},
            cleaning_report=cleaning_report,
        )
        model = evaluation.model
        model.metadata = _model_metadata(temporal, results, evaluation, settings)
        model.save(model_dir)
        if reports_dir is not None:
            _write_reports(
                results,
                model,
                trainval,
                test,
                reports_dir,
                drift_html=drift_html,
                shap_sample=shap_sample,
                seed=settings.random_seed,
            )
        if track:
            _log_to_mlflow(results, model_dir, reports_dir)
    return TrainingOutcome(model=model, results=results)
