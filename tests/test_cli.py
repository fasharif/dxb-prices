from __future__ import annotations

import json
from pathlib import Path

import pytest

from dxb_prices import cli, report
from dxb_prices.download import Manifest, ManifestEntry, sha256_of


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


def test_data_info_uses_the_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    touch_months(raw, ["2026-01", "2026-02"])
    manifest = Manifest(raw / "manifest.json")
    first = raw / "transactions_2026-01.csv"
    manifest.entries[first.name] = ManifestEntry(
        file=first.name,
        source="dld-open-data-export",
        url="u",
        period_start="2026-01-01",
        period_end="2026-01-31",
        downloaded_at="2026-09-26T08:00:00+00:00",
        bytes=first.stat().st_size,
        sha256=sha256_of(first),
        rows=0,
        final=True,
    )
    manifest.save()
    info = cli.data_info(raw, sorted(raw.glob("transactions_*.csv")))
    assert info["period_start"] == "2026-01-01" and info["downloaded_on"] == "2026-09-26"
    assert info["files"][1]["source"] == "not in manifest"
    assert len(info["files"][1]["sha256"]) == 64


def test_train_fixture_command_writes_a_model(tmp_path: Path) -> None:
    out = tmp_path / "model"
    assert cli.main(["train-fixture", "--model-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"model.lgb", "features.json", "metadata.json"}


def test_render_docs(tmp_path: Path) -> None:
    from tests.test_drift_and_report import results

    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(results()), encoding="utf-8")
    readme, card = tmp_path / "README.md", tmp_path / "model-card.md"
    for path in (readme, card):
        path.write_text(f"x\n{report.README_START}\n{report.README_END}\n", encoding="utf-8")
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
    assert "Test month by community data" in card.read_text(encoding="utf-8")
    assert (tmp_path / "metrics.md").read_text(encoding="utf-8").startswith("# Training")


def test_download_refuses_other_years(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="current calendar year"):
        cli.main(["download", "--months", "2020-01", "--raw-dir", str(tmp_path)])
