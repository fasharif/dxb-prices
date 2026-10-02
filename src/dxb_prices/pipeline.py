"""End-to-end training and evaluation.

Order of work, and which rows each step may see:

1. Split by month: oldest months train, the next month validates, the newest
   month tests.
2. Selection uses the training months to fit and the validation month to
   score: a small hyper-parameter search for the full model (with early
   stopping), then early stopping for the community-level model with the
   chosen settings. The validation month also sets each model's 80% range.
   The test month is not touched.
3. Both models are refitted on training + validation months with the chosen
   numbers of rounds, and both baselines are recomputed on the same rows.
   Only then is the test month scored, once.

The test month is scored the way the API answers. Each sale becomes the
request a user would send (community, project, size, rooms, off-plan, date)
and ``serving`` fills in the rest from the training rows. It is scored with
the project and without it. A further score uses DLD's recorded location
labels and freehold flag, which the API cannot know, as a reference.

A rolling-origin backtest repeats steps 2 and 3 with default settings for
every month that has enough history before it, to show how much the scores
move from one month to the next.

A simulated cold start refits the chosen models with each community in turn
cut to a few training sales (or none), and scores that community's test
sales, to show how the models do where the data is thin.
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

from dxb_prices import clean as clean_mod
from dxb_prices import features, metrics, report, schema, serving, split
from dxb_prices.baseline import CommunityMedianBaseline, ProjectMedianBaseline
from dxb_prices.config import Settings
from dxb_prices.features import COMMUNITY, FULL, VARIANT_FEATURES, FeatureSpec
from dxb_prices.model import BASE_PARAMS, SEARCH_SPACE, PriceModel, log_price_per_sqm, train_booster

log = logging.getLogger(__name__)

# Every estimator scored on the validation and test months, in report order.
ESTIMATORS: tuple[str, ...] = (
    "model",  # LightGBM as served, project given
    "model_no_project",  # LightGBM as served, project not given (community-level model)
    "project_baseline",  # project median per sqm, else community median
    "baseline",  # community median per sqm (the brief's baseline)
    "model_recorded",  # full model with DLD's recorded location labels (reference)
)
# Estimators that come with an 80% range.
RANGED: tuple[str, ...] = ("model", "model_no_project")
# Estimators shown in the error analysis (the recorded-labels reference is left out).
SEGMENTED: tuple[str, ...] = ("model", "model_no_project", "project_baseline", "baseline")


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


# --- fitting --------------------------------------------------------------------


@dataclass
class _Fit:
    model: PriceModel
    rounds: dict[str, int]
    community_baseline: CommunityMedianBaseline
    project_baseline: ProjectMedianBaseline
    rows: pd.DataFrame


def _prepare(rows: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, FeatureSpec]:
    fit_rows = split.trim_training(rows, settings.trim)
    return fit_rows, features.fit(fit_rows, settings.features)


def _booster(
    fit_rows: pd.DataFrame,
    spec: FeatureSpec,
    variant: str,
    params: dict[str, Any],
    *,
    valid: pd.DataFrame | None = None,
    rounds: int | None = None,
) -> lgb.Booster:
    columns = VARIANT_FEATURES[variant]
    x = features.transform(fit_rows, spec, columns)
    y = log_price_per_sqm(fit_rows)
    if valid is not None:
        return train_booster(
            x,
            y,
            params,
            x_valid=features.transform(valid, spec, columns),
            y_valid=log_price_per_sqm(valid),
        )
    if rounds is None:
        raise ValueError("a booster needs a validation month or a fixed number of rounds")
    return train_booster(x, y, params, num_boost_round=rounds)


def _rounds(booster: lgb.Booster) -> int:
    return int(booster.best_iteration) or int(booster.current_iteration())


def _assemble(
    fit_rows: pd.DataFrame, spec: FeatureSpec, boosters: dict[str, lgb.Booster], settings: Settings
) -> _Fit:
    return _Fit(
        model=PriceModel(boosters, spec, {}),
        rounds={variant: _rounds(b) for variant, b in boosters.items()},
        community_baseline=CommunityMedianBaseline.fit(fit_rows),
        project_baseline=ProjectMedianBaseline.fit(fit_rows, settings.baseline.min_project_rows),
        rows=fit_rows,
    )


def _refit(
    rows: pd.DataFrame, settings: Settings, params: dict[str, Any], rounds: dict[str, int]
) -> _Fit:
    fit_rows, spec = _prepare(rows, settings)
    boosters = {
        variant: _booster(fit_rows, spec, variant, params, rounds=rounds[variant])
        for variant in VARIANT_FEATURES
    }
    return _assemble(fit_rows, spec, boosters, settings)


def _search(
    fit_rows: pd.DataFrame,
    spec: FeatureSpec,
    valid: pd.DataFrame,
    *,
    candidates: tuple[dict[str, Any], ...],
    track: bool,
) -> tuple[dict[str, Any], lgb.Booster, list[dict[str, Any]]]:
    """Choose the full model's settings by MdAPE on the validation month."""
    x_va = features.transform(valid, spec)
    area = valid["area_sqm"].to_numpy(np.float64)
    trials: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], lgb.Booster] | None = None
    for i, params in enumerate(candidates):
        booster = _booster(fit_rows, spec, FULL, params, valid=valid)
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
    if best is None:
        raise ValueError("the search needs at least one candidate")
    return best[1], best[2], trials


# --- scoring --------------------------------------------------------------------


def score_period(fit: _Fit, frame: pd.DataFrame) -> pd.DataFrame:
    """Each estimator's price for every row of ``frame`` (plus 80% ranges once they exist)."""
    model = fit.model
    with_ranges = "residual_quantiles" in model.metadata
    with_project = serving.model_rows(
        serving.requests_from_sales(frame, with_project=True), model.spec
    )
    without = serving.model_rows(serving.requests_from_sales(frame, with_project=False), model.spec)
    served = serving.estimate(model, with_project, with_range=with_ranges)
    community = serving.estimate(model, without, with_range=with_ranges)
    scored = frame.copy()
    scored["variant"] = with_project["variant"]
    scored["pred_model"] = served["estimate"]
    scored["pred_model_no_project"] = community["estimate"]
    scored["pred_model_recorded"] = model.predict(frame, FULL)
    scored["pred_project_baseline"] = fit.project_baseline.predict(frame)
    scored["pred_baseline"] = fit.community_baseline.predict(frame)
    if with_ranges:
        for name, part in (("model", served), ("model_no_project", community)):
            scored[f"low_{name}"] = part["low"]
            scored[f"high_{name}"] = part["high"]
    return scored


def _scores(scored: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        name: metrics.score(scored["price_aed"], scored[f"pred_{name}"]).as_dict()
        for name in ESTIMATORS
    }


def _in_range(scored: pd.DataFrame, name: str) -> pd.Series:
    inside: pd.Series = (scored["price_aed"] >= scored[f"low_{name}"]) & (
        scored["price_aed"] <= scored[f"high_{name}"]
    )
    return inside


def _residual_quantiles(scored: pd.DataFrame) -> dict[str, dict[str, float]]:
    """10th and 90th percentiles of log(price / estimate) for each model, on the validation month.

    The full model's come from the sales it would serve (project known); the
    community-level model's from every sale, scored without the project.
    """

    def quantiles(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
        residuals = np.log(actual.to_numpy(np.float64) / predicted.to_numpy(np.float64))
        return {
            "q10": float(np.quantile(residuals, 0.10)),
            "q90": float(np.quantile(residuals, 0.90)),
        }

    community = quantiles(scored["price_aed"], scored["pred_model_no_project"])
    known = scored["variant"] == FULL
    full = quantiles(scored.loc[known, "price_aed"], scored.loc[known, "pred_model"])
    return {FULL: full if known.any() else community, COMMUNITY: community}


def _community_segment(frame: pd.DataFrame, rows: dict[str, int], thin: int) -> pd.Series:
    counts = frame["community"].astype("object").map(rows).fillna(0)
    order = [
        "new (no training sales)",
        f"thin (1-{thin - 1} training sales)",
        f"established ({thin}+ training sales)",
    ]
    labels = np.where(counts == 0, order[0], np.where(counts < thin, order[1], order[2]))
    return pd.Series(pd.Categorical(labels, categories=order), index=frame.index)


def _project_segment(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """How well the full model knows each sale's building, as the API would see it."""
    order = [
        f"own level ({spec.min_rows_project}+ training sales)",
        f"grouped as other (1-{spec.min_rows_project - 1} training sales)",
        "new (no training sales in its community)",
        "not recorded in the sale",
    ]
    labels = []
    for community, project in zip(
        frame["community"].astype("object"), frame["project"].astype("object"), strict=True
    ):
        if not isinstance(project, str):
            labels.append(order[3])
        elif not spec.has_project(community, project):
            labels.append(order[2])
        elif spec.project_has_own_level(community, project):
            labels.append(order[0])
        else:
            labels.append(order[1])
    return pd.Series(pd.Categorical(labels, categories=order), index=frame.index)


def _profile(part: pd.DataFrame) -> dict[str, Any]:
    """What a segment is made of, so a small or lopsided segment is visible."""
    projects = part["project"].astype("object").fillna("(no project)").value_counts()
    return {
        "communities": int(part["community"].nunique()),
        "largest_project_share": float(projects.iloc[0] / len(part)),
    }


# Error-analysis dimensions, in report order: (name, column added by segment_scores).
SEGMENT_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("registration", "registration"),
    ("price band", "price_band"),
    ("estimated price band", "estimate_band"),
    ("community data", "community_data"),
    ("project data", "project_data"),
    ("rooms", "rooms_segment"),
)


def segment_scores(
    scored: pd.DataFrame, spec: FeatureSpec, settings: Settings
) -> list[dict[str, Any]]:
    """Scores per segment. Price bands come twice: by the recorded price and by the estimate.

    Banding by the recorded price puts the sales that came in above their
    estimate into the higher bands, which can make the top band look worse
    than it is for a user, who only knows the estimate.
    """
    frame = scored.copy()
    frame["registration"] = pd.Categorical(
        np.where(frame["is_off_plan"].astype("boolean").fillna(False), "off-plan", "ready"),
        categories=["off-plan", "ready"],
    )
    bands = settings.segments.price_bands
    frame["price_band"] = metrics.price_band_labels(frame["price_aed"], bands)
    frame["estimate_band"] = metrics.price_band_labels(frame["pred_model"], bands)
    frame["community_data"] = _community_segment(
        frame, spec.community_rows, settings.segments.thin_community_rows
    )
    frame["project_data"] = _project_segment(frame, spec)
    frame["rooms_segment"] = pd.Categorical(frame["rooms"], categories=clean_mod.ROOM_LEVELS)
    columns = {name: f"pred_{name}" for name in SEGMENTED}
    coverage = {name: _in_range(frame, name) for name in RANGED if f"low_{name}" in frame}
    out: list[dict[str, Any]] = []
    for dimension, col in SEGMENT_DIMENSIONS:
        table = metrics.segment_table(frame, col, columns)
        profiles = {
            str(seg): _profile(part) for seg, part in frame.groupby(col, observed=True, sort=True)
        }
        for rec in table.to_dict(orient="records"):
            name, segment = str(rec["estimator"]), str(rec["segment"])
            if name in coverage:
                inside = coverage[name].groupby(frame[col], observed=True).mean()
                rec["range_coverage"] = float(inside[segment])
            out.append(
                {"dimension": dimension, **{str(k): v for k, v in rec.items()}} | profiles[segment]
            )
    return out


# --- selection, refit, test ---------------------------------------------------------


@dataclass
class _Selection:
    params: dict[str, Any]
    rounds: dict[str, int]
    trials: list[dict[str, Any]]
    residual_quantiles: dict[str, dict[str, float]]
    scores: dict[str, dict[str, Any]]
    rows_after_trim: int


def _select(
    train: pd.DataFrame, valid: pd.DataFrame, settings: Settings, *, search: bool, track: bool
) -> _Selection:
    """Fit on the training months; choose settings and set the 80% ranges on validation."""
    fit_rows, spec = _prepare(train, settings)
    params, full, trials = _search(
        fit_rows, spec, valid, candidates=SEARCH_SPACE if search else ({},), track=track
    )
    community = _booster(fit_rows, spec, COMMUNITY, params, valid=valid)
    fit = _assemble(fit_rows, spec, {FULL: full, COMMUNITY: community}, settings)
    scored = score_period(fit, valid)
    return _Selection(
        params=params,
        rounds=fit.rounds,
        trials=trials,
        residual_quantiles=_residual_quantiles(scored),
        scores=_scores(scored),
        rows_after_trim=len(fit_rows),
    )


@dataclass
class _Evaluation:
    fit: _Fit
    scored: pd.DataFrame
    scores: dict[str, dict[str, Any]]
    coverage: dict[str, float]
    segments: list[dict[str, Any]]


def _refit_and_test(
    trainval: pd.DataFrame, test: pd.DataFrame, settings: Settings, selection: _Selection
) -> _Evaluation:
    """Refit on training + validation months, then score the test month once."""
    fit = _refit(trainval, settings, selection.params, selection.rounds)
    fit.model.metadata = {"residual_quantiles": selection.residual_quantiles}
    scored = score_period(fit, test)
    return _Evaluation(
        fit=fit,
        scored=scored,
        scores=_scores(scored),
        coverage={name: float(_in_range(scored, name).mean()) for name in RANGED},
        segments=segment_scores(scored, fit.model.spec, settings),
    )


def _community_groups(communities: list[str], groups: int, seed: int) -> list[list[str]]:
    order = np.random.default_rng(seed).permutation(len(communities))
    n = max(1, min(groups, len(communities)))
    return [sorted(communities[j] for j in order[i::n]) for i in range(n)]


def cut_communities(
    rows: pd.DataFrame, communities: set[str], kept: int, seed: int
) -> pd.DataFrame:
    """Keep at most ``kept`` randomly chosen rows of each community in ``communities``."""
    inside = rows["community"].astype("object").isin(communities)
    draw = pd.Series(np.random.default_rng(seed).random(len(rows)), index=rows.index)
    rank = draw[inside].groupby(rows.loc[inside, "community"].astype("object")).rank(method="first")
    keep = ~inside
    keep.loc[rank.index] = rank <= kept
    cut: pd.DataFrame = rows.loc[keep]
    return cut


def cold_start(
    trainval: pd.DataFrame, test: pd.DataFrame, settings: Settings, selection: _Selection
) -> list[dict[str, Any]]:
    """Score each test sale with models refitted as if its community had little or no data.

    The test month's communities are split into random groups. For each group
    and each number of kept sales, both models and both baselines are refitted
    on the final training rows with that group's communities cut down (other
    communities keep all their rows), using the chosen settings, rounds and
    80% ranges; the group's test sales are then scored as served.
    """
    rules = settings.cold_start
    communities = sorted(str(c) for c in test["community"].dropna().unique())
    groups = _community_groups(communities, rules.groups, settings.random_seed)
    out: list[dict[str, Any]] = []
    for kept in rules.kept_rows:
        parts = []
        for group in groups:
            held = set(group)
            rows = cut_communities(trainval, held, kept, settings.random_seed)
            fit = _refit(rows, settings, selection.params, selection.rounds)
            fit.model.metadata = {"residual_quantiles": selection.residual_quantiles}
            parts.append(score_period(fit, test[test["community"].astype("object").isin(held)]))
        scored = pd.concat(parts)
        log.info(
            "cold start with %d training sales per community: MdAPE %.4f with project (%d rows)",
            kept,
            metrics.mdape(scored["price_aed"], scored["pred_model"]),
            len(scored),
        )
        out.append(
            {
                "kept_rows": kept,
                "groups": len(groups),
                "communities": len(communities),
                "rows": len(scored),
                "rows_full_model": int((scored["variant"] == FULL).sum()),
                "scores": {
                    name: metrics.score(scored["price_aed"], scored[f"pred_{name}"]).as_dict()
                    for name in SEGMENTED
                },
                "range_coverage": {name: float(_in_range(scored, name).mean()) for name in RANGED},
            }
        )
    return out


def backtest(cleaned: pd.DataFrame, settings: Settings) -> list[dict[str, Any]]:
    """Rolling origin: every month with enough history is tested once, with default settings.

    For test month T the models train on the months before T - 1, stop early on
    T - 1, are refitted on everything before T and are then scored on T. No
    search runs here, so no setting is chosen with a later month's data.
    """
    months = sorted(cleaned["month"].unique())
    rules = settings.split
    folds: list[dict[str, Any]] = []
    for i in range(rules.min_train_months + rules.valid_months, len(months)):
        train_months = months[: i - rules.valid_months]
        valid_months = months[i - rules.valid_months : i]
        train = cleaned[cleaned["month"].isin(train_months)]
        valid = cleaned[cleaned["month"].isin(valid_months)]
        test = cleaned[cleaned["month"] == months[i]]
        fit_rows, spec = _prepare(train, settings)
        boosters = {v: _booster(fit_rows, spec, v, {}, valid=valid) for v in VARIANT_FEATURES}
        rounds = {v: _rounds(b) for v, b in boosters.items()}
        final = _refit(pd.concat([train, valid]), settings, {}, rounds)
        scores = _scores(score_period(final, test))
        log.info(
            "backtest %s: MdAPE %.4f with project, %.4f without (%d rows)",
            months[i],
            scores["model"]["mdape"],
            scores["model_no_project"]["mdape"],
            len(test),
        )
        folds.append(
            {
                "test_month": str(months[i]),
                "train_months": [str(m) for m in train_months],
                "valid_months": [str(m) for m in valid_months],
                "rows": len(test),
                "rounds": rounds,
                "scores": scores,
            }
        )
    return folds


# --- results and reports ---------------------------------------------------------


def _results(
    temporal: split.TemporalSplit,
    rows: dict[str, int],
    selection: _Selection,
    evaluation: _Evaluation,
    *,
    folds: list[dict[str, Any]],
    cold: list[dict[str, Any]],
    command: str,
    data_info: dict[str, Any],
    cleaning_report: clean_mod.CleaningReport | None,
) -> dict[str, Any]:
    fit = evaluation.fit
    spec = fit.model.spec
    known = int((evaluation.scored["variant"] == FULL).sum())
    pairs = fit.rows.dropna(subset=["project"])[["community", "project"]].drop_duplicates()
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
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
                "final_fit": len(fit.rows),
            },
            "final_fit_communities": int(fit.rows["community"].nunique()),
            # A project is a name within a community (features.project_key).
            "final_fit_projects": len(pairs),
            "communities_with_own_level": len(spec.levels["community"]) - 2,
            "projects_with_own_level": len(spec.levels["project"]) - 2,
        },
        "search": selection.trials,
        "chosen": {
            "params": BASE_PARAMS | selection.params,
            "num_boost_round": selection.rounds[FULL],
            "num_boost_round_community": selection.rounds[COMMUNITY],
        },
        "served": {
            "test_rows": len(evaluation.scored),
            "test_rows_project_known": known,
            "min_project_rows_baseline": fit.project_baseline.min_rows,
            "test_rows_the_api_would_refuse": serving.refusals(evaluation.scored, spec),
        },
        "validation": selection.scores,
        "test": evaluation.scores,
        "interval": {
            "nominal": 0.8,
            "quantiles": selection.residual_quantiles,
            "test_coverage": evaluation.coverage,
        },
        "segments": evaluation.segments,
        "backtest": folds,
        "cold_start": cold,
    }


def _model_metadata(
    temporal: split.TemporalSplit,
    results: dict[str, Any],
    evaluation: _Evaluation,
    settings: Settings,
) -> dict[str, Any]:
    model = evaluation.fit.model
    probe = evaluation.fit.rows.head(1)
    base = {
        variant: float(np.exp(model.contributions(probe, variant)["bias"].iloc[0]))
        for variant in VARIANT_FEATURES
    }
    return {
        "model_version": f"{temporal.test_months[-1]}-{datetime.now(UTC):%Y%m%d%H%M%S}",
        "trained_at": results["generated_at"],
        "target": "natural log of price per square metre",
        "trained_on_months": list(temporal.train_months + temporal.valid_months),
        "evaluated_on_months": list(temporal.test_months),
        "data_period_end": results["data"].get("period_end"),
        "residual_quantiles": results["interval"]["quantiles"],
        "base_per_sqm": base,
        "baseline": evaluation.fit.community_baseline.to_dict(),
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
    # Explain the full model on the sales it serves, with inputs built as the API builds them.
    served = serving.model_rows(serving.requests_from_sales(test, with_project=True), model.spec)
    known = served.loc[served["variant"] == FULL]
    importance, gap = explain.global_importance(model, known, sample=shap_sample, seed=seed)
    results["shap"] = {
        "rows": min(shap_sample, len(known)),
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
    for name, value in results["interval"]["test_coverage"].items():
        flat[f"test_{name}_interval80_coverage"] = value
    for fold in results["backtest"]:
        for name in SEGMENTED:
            flat[f"backtest_{fold['test_month']}_{name}_mdape"] = fold["scores"][name]["mdape"]
    for level in results["cold_start"]:
        for name in SEGMENTED:
            flat[f"cold_start_{level['kept_rows']}_{name}_mdape"] = level["scores"][name]["mdape"]
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
    run_backtest: bool = False,
    run_cold_start: bool = False,
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
    command = (
        "dxb-prices train"
        + ("" if search else " --no-search")
        + ("" if run_backtest else " --no-backtest")
        + ("" if run_cold_start else " --no-cold-start")
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
        folds = backtest(cleaned, settings) if run_backtest else []
        cold = cold_start(trainval, test, settings, selection) if run_cold_start else []
        results = _results(
            temporal,
            {"train": len(train), "valid": len(valid), "test": len(test)},
            selection,
            evaluation,
            folds=folds,
            cold=cold,
            command=command,
            data_info=data_info or {},
            cleaning_report=cleaning_report,
        )
        model = evaluation.fit.model
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
