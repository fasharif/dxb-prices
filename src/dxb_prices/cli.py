"""Command line entry point: ``dxb-prices <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dxb_prices import config

EXIT_NOT_ENOUGH_DATA = 3


def _raw_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("transactions_*.csv"))


def data_info(raw_dir: Path, files: list[Path]) -> dict[str, Any]:
    """Describe the snapshot a run used: files, sizes, checksums and period."""
    from dxb_prices.download import Manifest, sha256_of

    manifest = Manifest(raw_dir / "manifest.json")
    entries: list[dict[str, Any]] = []
    for path in files:
        e = manifest.entries.get(path.name)
        if e is not None and e.sha256 == sha256_of(path):
            entries.append(
                {
                    "file": e.file,
                    "sha256": e.sha256,
                    "bytes": e.bytes,
                    "rows": e.rows,
                    "period_start": e.period_start,
                    "period_end": e.period_end,
                    "downloaded_at": e.downloaded_at,
                    "source": e.source,
                }
            )
        else:
            entries.append(
                {
                    "file": path.name,
                    "sha256": sha256_of(path),
                    "bytes": path.stat().st_size,
                    "source": "not in manifest",
                }
            )
    known = [e for e in entries if "period_start" in e]
    return {
        "source": config.DLD_PAGE_URL,
        "files": entries,
        "period_start": min((e["period_start"] for e in known), default=None),
        "period_end": max((e["period_end"] for e in known), default=None),
        "downloaded_on": max((e["downloaded_at"][:10] for e in known), default=None),
        "raw_rows": sum(int(e.get("rows", 0)) for e in entries),
    }


def cmd_download(args: argparse.Namespace) -> int:
    from dxb_prices import download

    if args.source == "dubai-data-sample":
        download.download_dubai_data_sample()
        return 0
    today = date.today()
    months = (
        [download.Month.parse(m) for m in args.months]
        if args.months
        else download.available_months(today, include_partial=args.include_partial)
    )
    if not months:
        print(
            "No complete month is available yet this year; use --include-partial.", file=sys.stderr
        )
        return EXIT_NOT_ENOUGH_DATA
    download.download_months(months, Path(args.raw_dir), today=today, force=args.force)
    return 0


def cmd_check_data(args: argparse.Namespace) -> int:
    """Exit 0 when the raw files hold enough months for a temporal split, 3 otherwise."""
    from dxb_prices import split
    from dxb_prices.download import Month

    months = sorted(
        Month.parse(p.stem.removeprefix("transactions_")).label
        for p in _raw_files(Path(args.raw_dir))
    )
    try:
        plan = split.plan(months, config.DEFAULT_SETTINGS.split)
    except split.InsufficientDataError as exc:
        print(f"not enough data: {exc}")
        return EXIT_NOT_ENOUGH_DATA
    print(f"ok: {len(months)} months; {plan.describe()}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from dxb_prices import pipeline

    raw_dir = Path(args.raw_dir)
    files = _raw_files(raw_dir)
    settings = config.DEFAULT_SETTINGS
    cleaned, cleaning_report, raw_rows = pipeline.load_and_clean(files, settings)
    info = data_info(raw_dir, files)
    info["raw_rows"] = raw_rows
    info["clean_rows"] = len(cleaned)
    reports_dir = Path(args.reports_dir)
    outcome = pipeline.train_and_evaluate(
        cleaned,
        settings,
        model_dir=Path(args.model_dir),
        reports_dir=reports_dir,
        search=not args.no_search,
        cleaning_report=cleaning_report,
        data_info=info,
        track=not args.no_tracking,
        tracking_uri=args.tracking_uri,
        drift_html=reports_dir / "drift" / f"drift_{cleaned['month'].max()}.html",
    )
    test = outcome.results["test"]
    print(json.dumps({"test": test, "model_dir": str(args.model_dir)}, indent=2))
    return 0


def cmd_train_fixture(args: argparse.Namespace) -> int:
    """Train a small model on the synthetic fixture (for tests, CI and a quick demo)."""
    from dxb_prices import fixture, pipeline

    cleaned, _, _ = pipeline.load_and_clean([Path(args.fixture)], fixture.SETTINGS)
    pipeline.train_and_evaluate(
        cleaned, fixture.SETTINGS, model_dir=Path(args.model_dir), reports_dir=None, search=False
    )
    print(f"synthetic-fixture model written to {args.model_dir}")
    return 0


def cmd_render_docs(args: argparse.Namespace) -> int:
    """Re-render reports/metrics.md and the results blocks in the README and model card."""
    from dxb_prices import report

    metrics_path = Path(args.metrics)
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_md = metrics_path.with_suffix(".md")
    report_md.write_text(report.metrics_markdown(results), encoding="utf-8")
    print(f"rendered {report_md}")
    for path, update in (
        (Path(args.readme), report.update_readme),
        (Path(args.model_card), report.update_model_card),
    ):
        path.write_text(update(path.read_text(encoding="utf-8"), results), encoding="utf-8")
        print(f"updated results block in {path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("dxb_prices.api.app:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dxb-prices", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download DLD monthly exports into the cache")
    p.add_argument(
        "--months",
        nargs="*",
        metavar="YYYY-MM",
        help="months to fetch (default: every complete month of this year)",
    )
    p.add_argument(
        "--include-partial", action="store_true", help="also fetch the current, incomplete month"
    )
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.add_argument("--raw-dir", default=str(config.RAW_DIR))
    p.add_argument(
        "--source",
        choices=("dld", "dubai-data-sample"),
        default="dld",
        help="dubai-data-sample fetches the 7,000-row public sample from data.dubai",
    )
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("check-data", help="exit 3 if there are too few months to train")
    p.add_argument("--raw-dir", default=str(config.RAW_DIR))
    p.set_defaults(func=cmd_check_data)

    p = sub.add_parser("train", help="clean, train, evaluate, explain and report")
    p.add_argument("--raw-dir", default=str(config.RAW_DIR))
    p.add_argument("--model-dir", default=str(config.MODEL_DIR))
    p.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    p.add_argument("--no-search", action="store_true", help="skip the hyper-parameter search")
    p.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    p.add_argument("--tracking-uri", default=None, help="MLflow URI (default: ./mlruns)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-fixture", help="train a small model on the synthetic fixture")
    p.add_argument(
        "--fixture",
        default=str(config.PROJECT_ROOT / "tests" / "fixtures" / "transactions_synthetic.csv"),
    )
    p.add_argument("--model-dir", default=str(config.ARTIFACTS_DIR / "fixture-model"))
    p.set_defaults(func=cmd_train_fixture)

    p = sub.add_parser("render-docs", help="re-render the report, README and model card results")
    p.add_argument("--metrics", default=str(config.REPORTS_DIR / "metrics.json"))
    p.add_argument("--readme", default=str(config.PROJECT_ROOT / "README.md"))
    p.add_argument("--model-card", default=str(config.PROJECT_ROOT / "docs" / "model-card.md"))
    p.set_defaults(func=cmd_render_docs)

    p = sub.add_parser("serve", help="run the API locally")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
