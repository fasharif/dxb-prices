"""Render training results (reports/metrics.json) as Markdown.

Everything here is generated from the results of one run, so the README,
the model card and reports/metrics.md cannot disagree with it.
"""

from __future__ import annotations

import math
import re
from typing import Any

README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
HEADLINE_START = "<!-- headline:start -->"
HEADLINE_END = "<!-- headline:end -->"

LABELS: dict[str, str] = {
    "model": "LightGBM, project given",
    "model_no_project": "LightGBM, no project (community-level model)",
    "project_baseline": "Baseline: project median",
    "baseline": "Baseline: community median",
    "model_recorded": "LightGBM with DLD's recorded location labels (reference)",
}
HEADLINE: tuple[str, ...] = tuple(LABELS)
SEGMENT_ESTIMATORS: tuple[str, ...] = ("model", "model_no_project", "project_baseline", "baseline")
# Error-analysis dimensions in report order (pipeline.SEGMENT_DIMENSIONS writes them).
SEGMENT_DIMENSIONS: tuple[str, ...] = (
    "registration",
    "price band",
    "estimated price band",
    "community data",
    "project data",
    "rooms",
)
SEGMENT_NOTES: dict[str, str] = {
    "price band": (
        "Bands by the recorded sale price. Banding by the outcome moves sales that sold above "
        "their estimate into higher bands and can make the top band look harder than it is, "
        "so the next table bands by the estimate instead."
    ),
    "estimated price band": (
        "Bands by the estimate with the project, which is what a user sees before a sale."
    ),
    "project data": (
        "How well the full model knows each sale's building. A project needs a minimum number "
        "of training sales in its community for a level of its own; rarer projects share one "
        "level, and a project with no training sales in its community (or none recorded) goes "
        "to the community-level model, so its two model columns are equal."
    ),
}
PROFILED: tuple[str, ...] = ("community data", "project data")
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _signed_pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def _aed(v: float) -> str:
    return f"{v:,.0f}"


def _months(months: list[str]) -> str:
    return ", ".join(months)


def _beats(m: dict[str, float], b: dict[str, float], name: str) -> str:
    wins = {
        "MdAPE": m["mdape"] < b["mdape"],
        "share within 10%": m["within_10pct"] > b["within_10pct"],
        "MAE": m["mae_aed"] < b["mae_aed"],
    }
    if all(wins.values()):
        return f"beats {name} on all three test metrics"
    if not any(wins.values()):
        return f"does not beat {name} on any test metric"
    better = " and ".join(k for k, v in wins.items() if v)
    worse = " and ".join(k for k, v in wins.items() if not v)
    return f"beats {name} on {better} but not on {worse}"


def building_share(test: dict[str, dict[str, float]]) -> str:
    """How much of the model's gain over the community median the project median already has."""
    base, proj, model = (test[k]["mdape"] for k in ("baseline", "project_baseline", "model"))
    numbers = (
        f"the project median alone moves MdAPE from {_pct(base)} to {_pct(proj)}, and the "
        f"model with the project reaches {_pct(model)}"
    )
    total, from_building = base - model, base - proj
    if total > 0 and from_building > total / 2:
        return (
            "Most of the gain over the community median comes from knowing the building: "
            f"{numbers}."
        )
    if total > 0 and from_building > 0:
        return (
            "Knowing the building gives less than half of the gain over the community median: "
            f"{numbers}."
        )
    return f"On MdAPE, {numbers}."


def verdict(test: dict[str, dict[str, float]]) -> str:
    """Plain statement of which baselines each served path beats on the test month."""
    community = _beats(test["model"], test["baseline"], "the community-median baseline")
    project = _beats(test["model"], test["project_baseline"], "the project-median baseline")
    if community.endswith("all three test metrics") and project.endswith("all three test metrics"):
        with_project = (
            "beats both the community-median and the project-median baseline on all three "
            "test metrics"
        )
    else:
        with_project = f"{community}, and {project}"
    without = _beats(test["model_no_project"], test["baseline"], "the community-median baseline")
    return (
        f"With the project, LightGBM {with_project}. Without the project, the community-level "
        f"model {without}. {building_share(test)}"
    )


def _coverage_line(results: dict[str, Any]) -> str:
    cover = results["interval"]["test_coverage"]
    return (
        f"The 80% range contained {_pct(cover['model'])} of test prices with the project "
        f"and {_pct(cover['model_no_project'])} without it (80% nominal)."
    )


def scores_table(results: dict[str, Any], stage: str) -> str:
    coverage = results["interval"]["test_coverage"] if stage == "test" else {}
    lines = [
        "| Estimator | Rows | MdAPE | Within 10% | MAE (AED) | Median error | In 80% range |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in HEADLINE:
        s = results[stage][name]
        cover = _pct(coverage[name]) if name in coverage else "n/a"
        lines.append(
            f"| {LABELS[name]} | {s['rows']:,} | {_pct(s['mdape'])} | "
            f"{_pct(s['within_10pct'])} | {_aed(s['mae_aed'])} | "
            f"{_signed_pct(s['median_error'])} | {cover} |"
        )
    return "\n".join(lines)


def refusal_note(served: dict[str, Any]) -> str:
    """Whether the API would have answered every test sale, and why not where it would not."""
    if "test_rows_the_api_would_refuse" not in served:
        return ""
    refused = {k: v for k, v in served["test_rows_the_api_would_refuse"].items() if v}
    if not refused:
        return "The API would have answered every test sale."
    total = sum(refused.values())
    counts = "; ".join(f"{reason}: {n:,}" for reason, n in refused.items())
    return (
        f"The API would refuse {total:,} test sale{'' if total == 1 else 's'} ({counts}); "
        "they are scored through the same code so that every estimator covers the same rows."
    )


def legend(results: dict[str, Any]) -> str:
    served = results["served"]
    refused = refusal_note(served)
    return (
        "*Project given*: the request a user sends (community, project, size, rooms, off-plan "
        "or ready, date); the nearest metro, mall and landmark and the freehold flag are "
        "filled in from the training data, as the API does. Sales whose project had no "
        "training sales in its community, or none recorded "
        f"({served['test_rows'] - served['test_rows_project_known']:,} of "
        f"{served['test_rows']:,}), get the community-level model, as they would from the API. "
        "*No project*: the same request without the project, answered by the community-level "
        "model, which was trained without the project and the location labels. "
        "*Project median*: the project's training median price per sqm when it has at least "
        f"{served['min_project_rows_baseline']} training sales, otherwise the community's, "
        "times the size. *Community median*: the community's training median price per sqm "
        "times the size (the baseline the brief asks for). *Recorded location labels*: the "
        "full model given DLD's own nearest metro, mall, landmark and freehold flag for each "
        "sale, which an API user cannot supply. Median error below zero means estimates run low."
        + (f" {refused}" if refused else "")
    )


def backtest_table(results: dict[str, Any]) -> str:
    lines = [
        "| Test month | Training months | Rows | MdAPE with project | MdAPE no project | "
        "Project median | Community median | Within 10% with project |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in results["backtest"]:
        s = fold["scores"]
        train = fold["train_months"] + fold["valid_months"]
        lines.append(
            f"| {fold['test_month']} | {train[0]} to {train[-1]} | {fold['rows']:,} | "
            f"{_pct(s['model']['mdape'])} | {_pct(s['model_no_project']['mdape'])} | "
            f"{_pct(s['project_baseline']['mdape'])} | {_pct(s['baseline']['mdape'])} | "
            f"{_pct(s['model']['within_10pct'])} |"
        )
    return "\n".join(lines)


def backtest_summary(results: dict[str, Any]) -> str:
    folds = results.get("backtest") or []
    if not folds:
        return ""

    def span(name: str) -> str:
        values = [f["scores"][name]["mdape"] for f in folds]
        return f"{_pct(min(values))} to {_pct(max(values))}"

    return (
        f"Rolling-origin backtest over {len(folds)} test months ({folds[0]['test_month']} to "
        f"{folds[-1]['test_month']}, default settings, each month scored by models trained "
        f"only on earlier months): MdAPE {span('model')} with the project and "
        f"{span('model_no_project')} without it, against {span('project_baseline')} for the "
        f"project median and {span('baseline')} for the community median."
    )


def _segment_rows(results: dict[str, Any], dimension: str) -> dict[str, dict[str, Any]]:
    by_segment: dict[str, dict[str, Any]] = {}
    for r in results["segments"]:
        if r["dimension"] == dimension:
            by_segment.setdefault(r["segment"], {})[r["estimator"]] = r
    return by_segment


def segments_table(results: dict[str, Any], dimension: str) -> str:
    lines = [
        "| Segment | Rows | MdAPE with project | MdAPE no project | Project median | "
        "Community median | Within 10% with project | In 80% range with project |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seg, est in _segment_rows(results, dimension).items():
        m = est["model"]
        cover = _pct(m["range_coverage"]) if "range_coverage" in m else "n/a"
        mdapes = " | ".join(_pct(est[name]["mdape"]) for name in SEGMENT_ESTIMATORS)
        lines.append(f"| {seg} | {m['rows']:,} | {mdapes} | {_pct(m['within_10pct'])} | {cover} |")
    return "\n".join(lines)


def profile_table(results: dict[str, Any], dimension: str) -> str:
    lines = [
        "| Segment | Rows | Communities | Largest single project |",
        "|---|---:|---:|---:|",
    ]
    for seg, est in _segment_rows(results, dimension).items():
        m = est["model"]
        lines.append(
            f"| {seg} | {m['rows']:,} | {m['communities']:,} | "
            f"{_pct(m['largest_project_share'])} of rows |"
        )
    return "\n".join(lines)


def _kept_label(kept: int) -> str:
    return "0: a new community (the API refuses these)" if kept == 0 else f"{kept:,}"


def cold_start_table(results: dict[str, Any]) -> str:
    levels = results.get("cold_start") or []
    lines = [
        "| Training sales kept per community | Rows | MdAPE with project | MdAPE no project | "
        "Project median | Community median | Within 10% with project | "
        "In 80% range with project |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    test, cover = results["test"], results["interval"]["test_coverage"]
    rows = [("all (the published models)", test, cover["model"])]
    rows += [
        (_kept_label(lv["kept_rows"]), lv["scores"], lv["range_coverage"]["model"])
        for lv in sorted(levels, key=lambda lv: -int(lv["kept_rows"]))
    ]
    for label, scores, coverage in rows:
        mdapes = " | ".join(_pct(scores[name]["mdape"]) for name in SEGMENT_ESTIMATORS)
        lines.append(
            f"| {label} | {scores['model']['rows']:,} | {mdapes} | "
            f"{_pct(scores['model']['within_10pct'])} | {_pct(coverage)} |"
        )
    return "\n".join(lines)


def cold_start_method(results: dict[str, Any]) -> str:
    levels = results.get("cold_start") or []
    if not levels:
        return ""
    first = levels[0]
    return (
        f"The {first['communities']:,} communities of the test month were split at random into "
        f"{first['groups']} groups. For each group, both models and both baselines were refitted "
        "on the final training rows with that group's communities cut to the stated number of "
        "randomly chosen sales (every other community kept all of its sales), with the chosen "
        "settings and 80% ranges, and the group's test sales were scored as served. A request "
        "for a community without training sales gets a 404 from the API; the last row shows "
        "what the models would have answered."
    )


def cold_start_summary(results: dict[str, Any]) -> str:
    levels = {int(lv["kept_rows"]): lv["scores"] for lv in results.get("cold_start") or []}
    thin = sorted(k for k in levels if k > 0)
    if not thin:
        return ""
    k, test = thin[0], results["test"]
    text = (
        f"Simulated cold start (each community in turn cut to {k:,} training sales, models "
        f"refitted): MdAPE {_pct(levels[k]['model']['mdape'])} with the project and "
        f"{_pct(levels[k]['model_no_project']['mdape'])} without it, against "
        f"{_pct(levels[k]['baseline']['mdape'])} for the community median (with all the data: "
        f"{_pct(test['model']['mdape'])}, {_pct(test['model_no_project']['mdape'])} and "
        f"{_pct(test['baseline']['mdape'])})."
    )
    if 0 in levels:
        text += (
            " The API refuses a community without training sales; had it answered, MdAPE would "
            f"have been {_pct(levels[0]['model']['mdape'])}."
        )
    return text


def _agreement(gap: float) -> str:
    who = "LightGBM's built-in TreeSHAP (used by the API) and the `shap` library"
    if gap == 0:
        return f"{who} give identical values on these rows."
    return f"{who} agree to within {gap:.1e} on these rows."


def _data_line(results: dict[str, Any]) -> str:
    data = results.get("data") or {}
    if not data:
        return "Data: not recorded (trained without a download manifest)."
    return (
        f"Data: DLD open data transaction export, sales registered {data.get('period_start')} to "
        f"{data.get('period_end')}, downloaded {data.get('downloaded_on')} (UTC) "
        f"({data.get('raw_rows', 0):,} raw rows in {len(data.get('files', []))} monthly files)."
    )


def snapshot_note(results: dict[str, Any], metrics_link: str) -> str:
    start = (results.get("data") or {}).get("period_start")
    if not start:
        return ""
    year = str(start)[:4]
    return (
        f"The DLD page only offers dates in the current calendar year, so this {year} "
        f"snapshot cannot be downloaded again with this tool after 31 December {year}. "
        f"The file sizes and SHA-256 checksums in [reports/metrics.md]({metrics_link}) "
        "identify it."
    )


def _month_name(month: str) -> str:
    year, number = month.split("-")
    return f"{_MONTHS[int(number) - 1]} {year}"


def headline(results: dict[str, Any]) -> str:
    """One sentence for the top of the README."""
    test, split = results["test"], results["split"]
    return (
        f"On the {split['rows']['test']:,} sales of {_month_name(split['test_months'][-1])}, "
        f"which the models had not seen, the median error was {_pct(test['model']['mdape'])} "
        f"with the building named in the request and {_pct(test['model_no_project']['mdape'])} "
        f"without it, against {_pct(test['baseline']['mdape'])} for the community's median "
        "price per square metre."
    )


def _intro(results: dict[str, Any]) -> str:
    split = results["split"]
    return (
        f"Test month {_months(split['test_months'])} ({split['rows']['test']:,} sales, scored "
        f"once, after the models were refitted on {split['train_months'][0]} to "
        f"{split['valid_months'][-1]}), each sale scored as the API would answer it:"
    )


def metrics_markdown(results: dict[str, Any]) -> str:
    env = results["environment"]
    split = results["split"]
    chosen = results["chosen"]
    parts = [
        "# Training and evaluation report",
        "",
        "Generated by `" + results["command"] + "`. Every number below comes from that run.",
        "",
        _data_line(results),
        "",
        f"Environment: Python {env['python']} on {env['platform']}, LightGBM {env['lightgbm']}, "
        f"pandas {env['pandas']}.",
        "",
        "## Headline: the test month",
        "",
        _intro(results),
        "",
        scores_table(results, "test"),
        "",
        legend(results),
        "",
        verdict(results["test"]) + " " + _coverage_line(results),
        "",
        'MdAPE is the median absolute percentage error. "Within 10%" is the share of estimates '
        "within 10% of the recorded price. MAE is the mean absolute error in dirhams. "
        "Median error is the median of (estimate - price) / price.",
        "",
    ]
    if results.get("backtest"):
        parts += [
            "## Rolling-origin backtest",
            "",
            backtest_summary(results),
            "",
            backtest_table(results),
            "",
        ]
    parts += [
        "## Validation month",
        "",
        f"{_months(split['valid_months'])}, scored by the models fitted on the training months "
        "only. This month chose the settings, stopped the boosting and set the 80% ranges, so "
        "these scores are optimistic; the test month above is the honest estimate.",
        "",
        scores_table(results, "validation"),
        "",
        "## Split",
        "",
        f"- Train: {_months(split['train_months'])} ({split['rows']['train']:,} rows; "
        f"{split['rows']['train_after_trim']:,} after trimming the price-per-sqm tails)",
        f"- Validation: {_months(split['valid_months'])} ({split['rows']['valid']:,} rows)",
        f"- Final fit on train + validation: {split['rows']['final_fit']:,} rows",
        f"- Test: {_months(split['test_months'])} ({split['rows']['test']:,} rows, not trimmed)",
        "",
        f"Chosen settings for the full model: `{chosen['params']['objective']}` loss, "
        f"{chosen['params']['num_leaves']} leaves, "
        f"min {chosen['params']['min_data_in_leaf']} rows per leaf, "
        f"{chosen['num_boost_round']} boosting rounds "
        f"(best of {len(results['search'])} settings on the validation month). The "
        f"community-level model uses the same settings with {chosen['num_boost_round_community']}"
        " rounds (early stopping on the validation month).",
        "",
    ]
    files = (results.get("data") or {}).get("files") or []
    if files:
        parts += [
            "## Data snapshot",
            "",
            "| File | Rows | Bytes | SHA-256 | Downloaded |",
            "|---|---:|---:|---|---|",
        ]
        for f in files:
            parts.append(
                f"| {f.get('file')} | {int(f.get('rows', 0)):,} | "
                f"{int(f.get('bytes', 0)):,} | `{f.get('sha256')}` | "
                f"{str(f.get('downloaded_at', ''))[:10]} |"
            )
        parts += [
            "",
            "The export can change after download (late registrations), so a later "
            "download of the same months may not match these checksums.",
            "",
        ]
    if results.get("cleaning"):
        parts += [
            "## Cleaning",
            "",
            "| Step | Rule | Rows in | Removed | Rows out |",
            "|---|---|---:|---:|---:|",
        ]
        for s in results["cleaning"]:
            parts.append(
                f"| {s['step']} | {s['rule']} | {s['rows_in']:,} | {s['removed']:,} | "
                f"{s['rows_out']:,} |"
            )
        parts.append("")
    parts += [
        "## Error analysis on the test month",
        "",
        "MdAPE of each estimator per segment, inputs as in the headline.",
        "",
    ]
    for dimension in SEGMENT_DIMENSIONS:
        if not _segment_rows(results, dimension):
            continue
        parts += [f"### By {dimension}", ""]
        if dimension in SEGMENT_NOTES:
            parts += [SEGMENT_NOTES[dimension], ""]
        parts += [segments_table(results, dimension), ""]
        if dimension in PROFILED:
            parts += [
                "What these segments contain (a segment dominated by one project or made of a "
                "few unusual sales says little about the segment in general):",
                "",
                profile_table(results, dimension),
                "",
            ]
    if results.get("cold_start"):
        parts += [
            "## Cold start: communities with little or no data",
            "",
            cold_start_method(results),
            "",
            cold_start_table(results),
            "",
            cold_start_summary(results),
            "",
        ]
    q = results["interval"]["quantiles"]

    def bounds(variant: str) -> str:
        low, high = (math.exp(q[variant][k]) - 1 for k in ("q10", "q90"))
        return f"{_signed_pct(low)} to {_signed_pct(high)}"

    parts += [
        "## Estimate range",
        "",
        "The API returns an 80% range: the 10th and 90th percentiles of each model's residuals "
        "on the validation month, applied around the estimate "
        f"({bounds('full')} for the full model, {bounds('community')} for the "
        "community-level model). It has the same relative width for every estimate. "
        + _coverage_line(results),
        "",
    ]
    if "shap" in results:
        shap = results["shap"]
        parts += [
            "## What drives the estimates (SHAP)",
            "",
            f"Mean absolute SHAP value per feature of the full model on {shap['rows']:,} test "
            "sales whose project it knows, with inputs built as the API builds them. "
            + _agreement(shap["max_abs_gap_shap_vs_lightgbm"]),
            "",
            "| Feature | Mean abs SHAP | Share |",
            "|---|---:|---:|",
            *[
                f"| {r['label']} | {r['mean_abs_shap']:.4f} | {_pct(r['share'])} |"
                for r in shap["importance"]
            ],
            "",
            "![Mean absolute SHAP value per feature](shap_importance.png)",
            "",
        ]
    if "drift" in results:
        d = results["drift"]
        parts += [
            "## Drift: newest month against the training data",
            "",
            f"Evidently `DataDriftPreset`, {d['current_month']} ({d['current_rows']:,} rows) "
            f"against {_months(d['reference_months'])} ({d['reference_rows']:,} rows). "
            f"Columns flagged as drifted: {int(d['drifted_count'] or 0)} of {len(d['columns'])}.",
            "",
            "| Column | Test | Value | Threshold | Drift |",
            "|---|---|---:|---:|---|",
            *[
                f"| {c['column']} | {c['method']} | {c['value']:.4f} | {c['threshold']:.2f} | "
                f"{'yes' if c['drifted'] else 'no'} |"
                for c in d["columns"]
            ],
            "",
            "The full HTML report is written to `reports/drift/` (not committed: it embeds "
            "distributions of the source data).",
            "",
        ]
    return "\n".join(parts)


def _block(lines: list[str]) -> str:
    return "\n".join([README_START, *lines, README_END])


def readme_block(results: dict[str, Any]) -> str:
    env = results["environment"]
    lines = [
        _data_line(results),
        "",
        _intro(results),
        "",
        scores_table(results, "test"),
        "",
        "<details>",
        "<summary>What each row means</summary>",
        "",
        legend(results),
        "",
        "</details>",
        "",
        verdict(results["test"]) + " " + _coverage_line(results),
        "",
    ]
    if results.get("backtest"):
        lines += [backtest_summary(results), ""]
    cold = cold_start_summary(results)
    if cold:
        lines += [cold, ""]
    note = snapshot_note(results, "reports/metrics.md")
    if note:
        lines += [note, ""]
    lines.append(
        f"Environment: Python {env['python']}, LightGBM {env['lightgbm']}. Reproduce with "
        f"`{results['command']}`; the validation month, backtest and error analysis are in "
        "[reports/metrics.md](reports/metrics.md)."
    )
    return _block(lines)


def model_card_block(results: dict[str, Any]) -> str:
    lines = [
        _data_line(results),
        "",
        _intro(results),
        "",
        scores_table(results, "test"),
        "",
        legend(results),
        "",
        verdict(results["test"]) + " " + _coverage_line(results),
        "",
    ]
    if results.get("backtest"):
        lines += [backtest_summary(results), ""]
    for dimension in SEGMENT_DIMENSIONS[:-1]:
        if not _segment_rows(results, dimension):
            continue
        lines += [f"Test month by {dimension}:", ""]
        if dimension in SEGMENT_NOTES:
            lines += [SEGMENT_NOTES[dimension], ""]
        lines += [segments_table(results, dimension), ""]
        if dimension in PROFILED:
            lines += [
                f"What the {dimension} segments contain:",
                "",
                profile_table(results, dimension),
                "",
            ]
    if results.get("cold_start"):
        lines += [
            "Simulated cold start:",
            "",
            cold_start_method(results),
            "",
            cold_start_table(results),
            "",
        ]
    note = snapshot_note(results, "../reports/metrics.md")
    if note:
        lines += [note, ""]
    if "drift" in results:
        d = results["drift"]
        drifted = [c["column"] for c in d["columns"] if c["drifted"]]
        lines += [
            f"Drift ({d['current_month']} against the training months): "
            + (f"flagged for {', '.join(drifted)}." if drifted else "no column flagged."),
            "",
        ]
    return _block(lines)


def replace_block(text: str, block: str, start: str = README_START, end: str = README_END) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"document has no {start} ... {end} block")
    return pattern.sub(lambda _: block, text)


def update_readme(readme: str, results: dict[str, Any]) -> str:
    top = "\n".join([HEADLINE_START, headline(results), HEADLINE_END])
    with_headline = replace_block(readme, top, HEADLINE_START, HEADLINE_END)
    return replace_block(with_headline, readme_block(results))


def update_model_card(card: str, results: dict[str, Any]) -> str:
    return replace_block(card, model_card_block(results))
