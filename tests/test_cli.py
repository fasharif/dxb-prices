from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from dxb_prices import cli, report
from dxb_prices.download import Manifest, ManifestEntry, sha256_of
from tests.conftest import FIXTURE_CSV


def touch_months(raw: Path, months: list[str]) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for m in months:
        (raw / f"transactions_{m}.csv").write_text(
            "TRANSACTION_NUMBER,TRANS_VALUE\n", encoding="utf-8"
        )


def test_check_data_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw"
    touch_months(raw, ["2026-01", "2026-02"])
    assert cli.main(["check-data", "--raw-dir", str(raw)]) == cli.EXIT_NOT_ENOUGH_DATA
    touch_months(raw, ["2026-03", "2026-04"])
    assert cli.main(["check-data", "--raw-dir", str(raw)]) == 0
    assert "test 2026-04" in capsys.readouterr().out


def entry_for(path: Path, *, final: bool) -> ManifestEntry:
    month = path.stem.removeprefix("transactions_")
    return ManifestEntry(
        file=path.name,
        source="dld-open-data-export",
        url="u",
        period_start=f"{month}-01",
        period_end=f"{month}-28",
        downloaded_at="2026-05-03T08:00:00+00:00",
        bytes=path.stat().st_size,
        sha256=sha256_of(path),
        rows=0,
        final=final,
    )


def test_check_data_refuses_a_provisional_newest_month(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    touch_months(raw, ["2026-01", "2026-02", "2026-03", "2026-04"])
    manifest = Manifest(raw / "manifest.json")
    newest = raw / "transactions_2026-04.csv"
    manifest.entries[newest.name] = entry_for(newest, final=False)
    manifest.save()
    assert cli.main(["check-data", "--raw-dir", str(raw)]) == cli.EXIT_NOT_ENOUGH_DATA
    assert "2026-04, was downloaded less than 7 days" in capsys.readouterr().out
    manifest.entries[newest.name] = entry_for(newest, final=True)
    manifest.save()
    assert cli.main(["check-data", "--raw-dir", str(raw)]) == 0


def test_download_in_january_finds_nothing_to_do_and_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The scheduled workflow runs `download` then `check-data`; January must not fail it.
    monkeypatch.setattr(cli, "today", lambda: date(2027, 1, 10))
    raw = tmp_path / "raw"
    assert cli.main(["download", "--raw-dir", str(raw)]) == 0
    assert "No complete month of 2027" in capsys.readouterr().out
    assert not raw.exists()
    raw.mkdir()
    assert cli.main(["check-data", "--raw-dir", str(raw)]) == cli.EXIT_NOT_ENOUGH_DATA


def test_train_command_runs_end_to_end(tmp_path: Path) -> None:
    raw, model, reports = tmp_path / "raw", tmp_path / "model", tmp_path / "reports"
    raw.mkdir()
    shutil.copy(FIXTURE_CSV, raw / "transactions_2026-04.csv")
    code = cli.main(
        [
            "train",
            "--raw-dir",
            str(raw),
            "--model-dir",
            str(model),
            "--reports-dir",
            str(reports),
            "--no-search",
            "--no-tracking",
            "--no-backtest",
            "--no-cold-start",
        ]
    )
    assert code == 0
    assert (model / "model.lgb").exists()
    results = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    assert results["split"]["test_months"] == ["2026-04"]
    assert results["command"] == "dxb-prices train --no-search --no-backtest --no-cold-start"
    assert results["backtest"] == [] and results["cold_start"] == []
    assert results["data"]["files"][0]["source"] == "not in manifest"
    assert (reports / "metrics.md").exists()
    assert (reports / "drift" / "drift_2026-04.html").exists()


def test_data_info_uses_the_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    touch_months(raw, ["2026-01", "2026-02"])
    manifest = Manifest(raw / "manifest.json")
    first = raw / "transactions_2026-01.csv"
    manifest.entries[first.name] = entry_for(first, final=True)
    manifest.save()
    info = cli.data_info(raw, sorted(raw.glob("transactions_*.csv")))
    assert info["period_start"] == "2026-01-01" and info["downloaded_on"] == "2026-05-03"
    assert info["files"][1]["source"] == "not in manifest"
    assert len(info["files"][1]["sha256"]) == 64


def test_train_fixture_command_writes_a_model(tmp_path: Path) -> None:
    out = tmp_path / "model"
    assert cli.main(["train-fixture", "--model-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {
        "model.lgb",
        "model_community.lgb",
        "features.json",
        "metadata.json",
    }


def test_render_docs(tmp_path: Path) -> None:
    from tests.test_drift_and_report import results

    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(results()), encoding="utf-8")
    readme, card = tmp_path / "README.md", tmp_path / "model-card.md"
    block = f"{report.README_START}\n{report.README_END}\n"
    readme.write_text(
        f"x\n{report.HEADLINE_START}\n{report.HEADLINE_END}\n{block}", encoding="utf-8"
    )
    card.write_text(f"x\n{block}", encoding="utf-8")
    assert (
        cli.main(
            [
                "render-docs",
                "--metrics",
                str(metrics),
                "--readme",
                str(readme),
                "--model-card",
                str(card),
            ]
        )
        == 0
    )
    assert "LightGBM" in readme.read_text(encoding="utf-8")
    assert "the median error was 5.0% with the building named" in readme.read_text(encoding="utf-8")
    assert "Test month by community data" in card.read_text(encoding="utf-8")
    assert (tmp_path / "metrics.md").read_text(encoding="utf-8").startswith("# Training")


def test_download_refuses_other_years(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["download", "--months", "2020-01", "--raw-dir", str(tmp_path)])
    assert code == cli.EXIT_ERROR
    assert "current calendar year" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["train"], "run `dxb-prices download` first"),
        (["download", "--months", "2026-13"], "month out of range"),
        (["render-docs", "--metrics", "{tmp}/metrics.json"], "run `dxb-prices train` first"),
    ],
)
def test_user_errors_print_a_message_instead_of_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    args = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    if args[0] in {"train", "download"}:
        args += ["--raw-dir", str(tmp_path / "raw")]
    assert cli.main(args) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert err.startswith(f"dxb-prices {args[0]}: error: ")
    assert message in err
    assert "Traceback" not in err


def test_train_with_too_few_months_says_how_many_it_needs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = FIXTURE_CSV.read_text(encoding="utf-8").splitlines()
    january = [lines[0], *(line for line in lines[1:] if '"2026-01-' in line)]
    (raw / "transactions_2026-01.csv").write_text("\n".join(january) + "\n", encoding="utf-8")
    code = cli.main(["train", "--raw-dir", str(raw), "--no-tracking", "--no-backtest"])
    assert code == cli.EXIT_ERROR
    assert "need at least 4 months" in capsys.readouterr().err
