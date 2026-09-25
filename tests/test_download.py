from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from dxb_prices import download, schema
from dxb_prices.download import DownloadError, Month

CSV = (
    "\ufeffTRANSACTION_NUMBER,INSTANCE_DATE,TRANS_VALUE,ACTUAL_AREA\n"
    '"1-1-2026","2026-01-02 10:00:00","1000000","70"\n'
    '"1-2-2026","2026-01-03 11:00:00","2000000","110"\n'
).encode()


class Recorder:
    """httpx handler that records requests and replays a list of responses."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def csv_response(body: bytes = CSV) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/csv"})


def client_for(handler: Recorder) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_available_months_are_the_complete_months_of_this_year() -> None:
    assert [m.label for m in download.available_months(date(2026, 9, 26))] == [
        f"2026-{m:02d}" for m in range(1, 9)
    ]
    assert download.available_months(date(2026, 9, 26), include_partial=True)[-1].label == "2026-09"
    assert download.available_months(date(2026, 1, 15)) == []


def test_months_outside_the_current_year_or_in_the_future_are_refused() -> None:
    with pytest.raises(ValueError, match="current calendar year"):
        download.check_in_window([Month(2025, 12)], date(2026, 9, 26))
    with pytest.raises(ValueError, match="future"):
        download.check_in_window([Month(2026, 10)], date(2026, 9, 26))


def test_month_parse() -> None:
    assert Month.parse("2026-03") == Month(2026, 3)
    assert Month(2026, 2).last_day == date(2026, 2, 28)
    for bad in ("2026-13", "March", "2026/03"):
        with pytest.raises(ValueError):
            Month.parse(bad)


def test_export_body_matches_the_page_request() -> None:
    body = download.export_body(date(2026, 3, 1), date(2026, 3, 31))
    params = body["parameters"]
    assert params["P_FROM_DATE"] == "03/01/2026"
    assert params["P_TO_DATE"] == "03/31/2026"
    assert params["P_TAKE"] == "-1"
    assert params["P_GROUP_ID"] == "" and params["P_PROP_TYPE_ID"] == ""
    assert body["command"] == "transactions"
    assert list(body["labels"]) == list(schema.EXPORT_COLUMNS)


def test_download_writes_file_and_manifest_with_checksum(tmp_path: Path) -> None:
    handler = Recorder(csv_response())
    entries = download.download_months(
        [Month(2026, 1)],
        tmp_path,
        today=date(2026, 9, 26),
        client=client_for(handler),
        sleep=lambda _: None,
    )
    path = tmp_path / "transactions_2026-01.csv"
    assert path.read_bytes() == CSV
    entry = entries[0]
    assert entry.sha256 == hashlib.sha256(CSV).hexdigest()
    assert entry.bytes == len(CSV)
    assert entry.rows == 2
    assert entry.final is True
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["transactions_2026-01.csv"]["sha256"] == entry.sha256
    sent = json.loads(handler.requests[0].content)
    assert sent["parameters"]["P_FROM_DATE"] == "01/01/2026"


def test_final_months_come_from_the_cache(tmp_path: Path) -> None:
    today = date(2026, 9, 26)
    download.download_months(
        [Month(2026, 1)],
        tmp_path,
        today=today,
        client=client_for(Recorder(csv_response())),
        sleep=lambda _: None,
    )
    handler = Recorder(csv_response(b"should not be requested"))
    download.download_months(
        [Month(2026, 1)], tmp_path, today=today, client=client_for(handler), sleep=lambda _: None
    )
    assert handler.requests == []


def test_provisional_months_and_corrupted_files_are_fetched_again(tmp_path: Path) -> None:
    # Downloaded three days after the month ended: provisional, so refreshed next time.
    download.download_months(
        [Month(2026, 8)],
        tmp_path,
        today=date(2026, 9, 3),
        client=client_for(Recorder(csv_response())),
        sleep=lambda _: None,
    )
    handler = Recorder(csv_response())
    download.download_months(
        [Month(2026, 8)],
        tmp_path,
        today=date(2026, 9, 20),
        client=client_for(handler),
        sleep=lambda _: None,
    )
    assert len(handler.requests) == 1
    # A final month whose file no longer matches its checksum is fetched again.
    (tmp_path / "transactions_2026-08.csv").write_bytes(CSV.replace(b"2000000", b"2000001"))
    handler = Recorder(csv_response())
    download.download_months(
        [Month(2026, 8)],
        tmp_path,
        today=date(2026, 9, 20),
        client=client_for(handler),
        sleep=lambda _: None,
    )
    assert len(handler.requests) == 1


def test_force_ignores_the_cache(tmp_path: Path) -> None:
    today = date(2026, 9, 26)
    download.download_months(
        [Month(2026, 1)],
        tmp_path,
        today=today,
        client=client_for(Recorder(csv_response())),
        sleep=lambda _: None,
    )
    handler = Recorder(csv_response())
    download.download_months(
        [Month(2026, 1)],
        tmp_path,
        today=today,
        force=True,
        client=client_for(handler),
        sleep=lambda _: None,
    )
    assert len(handler.requests) == 1


def test_html_instead_of_csv_is_reported_clearly(tmp_path: Path) -> None:
    page = httpx.Response(
        200, content=b"<!DOCTYPE html><html>challenge</html>", headers={"content-type": "text/html"}
    )
    with pytest.raises(DownloadError, match="HTML page instead of CSV"):
        download.download_months(
            [Month(2026, 1)],
            tmp_path,
            today=date(2026, 9, 26),
            client=client_for(Recorder(page)),
            sleep=lambda _: None,
        )
    assert not (tmp_path / "transactions_2026-01.csv").exists()


def test_csv_without_expected_columns_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="missing expected columns"):
        download.download_months(
            [Month(2026, 1)],
            tmp_path,
            today=date(2026, 9, 26),
            client=client_for(Recorder(csv_response(b"A,B\n1,2\n"))),
            sleep=lambda _: None,
        )


def test_server_errors_are_retried_with_backoff(tmp_path: Path) -> None:
    waits: list[float] = []
    handler = Recorder(httpx.Response(503), httpx.ConnectError("reset"), csv_response())
    download.download_months(
        [Month(2026, 1)],
        tmp_path,
        today=date(2026, 9, 26),
        client=client_for(handler),
        sleep=waits.append,
        attempts=3,
    )
    assert len(handler.requests) == 3
    assert waits == [5.0, 10.0]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    handler = Recorder(httpx.Response(400))
    with pytest.raises(DownloadError, match="HTTP 400"):
        download.download_months(
            [Month(2026, 1)],
            tmp_path,
            today=date(2026, 9, 26),
            client=client_for(handler),
            sleep=lambda _: None,
        )
    assert len(handler.requests) == 1


def test_polite_pause_between_months(tmp_path: Path) -> None:
    waits: list[float] = []
    download.download_months(
        [Month(2026, 1), Month(2026, 2)],
        tmp_path,
        today=date(2026, 9, 26),
        client=client_for(Recorder(csv_response())),
        sleep=waits.append,
        pause_seconds=3.0,
    )
    assert waits == [3.0]


def test_dubai_data_sample_is_saved_as_csv(tmp_path: Path) -> None:
    payload = {
        "success": True,
        "data": [
            {
                "transaction_id": "1-11-2020-1",
                "instance_date": "2020-01-02",
                "actual_worth": 900000,
            },
            {
                "transaction_id": "1-11-2025-9",
                "instance_date": "2025-05-06",
                "actual_worth": 1200000,
            },
        ],
    }
    client = client_for(Recorder(httpx.Response(200, json=payload)))
    entry = download.download_dubai_data_sample(tmp_path, client=client)
    text = (tmp_path / entry.file).read_text(encoding="utf-8")
    assert text.splitlines()[0] == "transaction_id,instance_date,actual_worth"
    assert entry.rows == 2
    assert (entry.period_start, entry.period_end) == ("2020-01-02", "2025-05-06")


def test_dubai_data_sample_failure_is_reported(tmp_path: Path) -> None:
    client = client_for(Recorder(httpx.Response(200, json={"success": False, "message": "no"})))
    with pytest.raises(DownloadError, match="no rows"):
        download.download_dubai_data_sample(tmp_path, client=client)
