"""Download DLD transaction exports month by month, with a local cache.

The DLD open data page offers a "Download as CSV" button for its transaction
search. This module sends the same request the page sends, one calendar month
at a time, and keeps a manifest with the size and SHA-256 of every file so a
training run can say exactly which snapshot it used.

The page's date picker only offers dates in the current calendar year, so the
downloader stays inside that window.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from dxb_prices import config
from dxb_prices.errors import UserFacingError
from dxb_prices.schema import EXPORT_COLUMNS

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset({"TRANSACTION_NUMBER", "INSTANCE_DATE", "TRANS_VALUE", "ACTUAL_AREA"})

MANIFEST_NAME = "manifest.json"
# Late registrations can still arrive shortly after a month ends. A month
# downloaded within this many days of its end is re-fetched on the next run.
PROVISIONAL_DAYS = 7
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


MANUAL_DOWNLOAD = (
    f"Download the month by hand from {config.DLD_PAGE_URL} and save it under data/raw/dld/ "
    "with the name transactions_YYYY-MM.csv"
)


class DownloadError(UserFacingError, RuntimeError):
    """The source did not return a usable CSV."""


class MonthError(UserFacingError, ValueError):
    """A month argument that is malformed or outside what the DLD page offers."""


@dataclass(frozen=True, order=True)
class Month:
    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def first_day(self) -> date:
        return date(self.year, self.month, 1)

    @property
    def last_day(self) -> date:
        return date(self.year, self.month, calendar.monthrange(self.year, self.month)[1])

    @classmethod
    def parse(cls, text: str) -> Month:
        try:
            year_s, month_s = text.split("-")
            month = cls(int(year_s), int(month_s))
        except ValueError as exc:
            raise MonthError(f"expected a month as YYYY-MM, got {text!r}") from exc
        if not 1 <= month.month <= 12:
            raise MonthError(f"month out of range in {text!r}")
        return month


def available_months(today: date, include_partial: bool = False) -> list[Month]:
    """Months the portal offers: January of this year up to the last complete month."""
    last = today.month if include_partial else today.month - 1
    return [Month(today.year, m) for m in range(1, last + 1)]


def check_in_window(months: Iterable[Month], today: date) -> None:
    for m in months:
        if m.year != today.year:
            raise MonthError(
                f"{m.label} is outside {today.year}: the DLD open data page only offers "
                "dates in the current calendar year, so this tool does not request others"
            )
        if m.first_day > today:
            raise MonthError(f"{m.label} is in the future")


def export_body(start: date, end: date) -> dict[str, Any]:
    """The JSON body the DLD page posts for a CSV export (all groups, all types)."""
    return {
        "parameters": {
            "P_FROM_DATE": start.strftime("%m/%d/%Y"),
            "P_TO_DATE": end.strftime("%m/%d/%Y"),
            "P_GROUP_ID": "",
            "P_IS_OFFPLAN": "",
            "P_IS_FREE_HOLD": "",
            "P_AREA_ID": "",
            "P_USAGE_ID": "",
            "P_PROP_TYPE_ID": "",
            "P_TAKE": "-1",
            "P_SKIP": "",
            "P_SORT": "TRANSACTION_NUMBER_ASC",
        },
        "command": "transactions",
        "labels": {c: c for c in EXPORT_COLUMNS},
    }


@dataclass
class ManifestEntry:
    file: str
    source: str
    url: str
    period_start: str
    period_end: str
    downloaded_at: str
    bytes: int
    sha256: str
    rows: int
    final: bool


class Manifest:
    """JSON record of every cached file: where it came from and its checksum."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, ManifestEntry] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.entries = {k: ManifestEntry(**v) for k, v in raw.get("files", {}).items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"files": {k: asdict(v) for k, v in sorted(self.entries.items())}}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(content: bytes) -> tuple[list[str], int]:
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader, [])
        return header, sum(1 for _ in reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DownloadError(
            f"the portal returned something that is not a UTF-8 CSV ({exc}). {MANUAL_DOWNLOAD}"
        ) from exc


def validate_csv(content: bytes, content_type: str) -> tuple[list[str], int]:
    """Check that a response is the expected CSV, not an error or challenge page."""
    if "html" in content_type.lower() or content.lstrip()[:15].lower().startswith(
        b"<!doctype html"
    ):
        raise DownloadError(
            "the portal returned an HTML page instead of CSV; it may be showing a "
            f"challenge or maintenance page. {MANUAL_DOWNLOAD}"
        )
    header, rows = _count_csv_rows(content)
    missing = REQUIRED_COLUMNS - set(header)
    if missing:
        raise DownloadError(f"CSV is missing expected columns: {sorted(missing)}")
    return header, rows


def _post_with_retries(
    client: httpx.Client,
    url: str,
    body: dict[str, Any],
    attempts: int,
    sleep: Callable[[float], None],
) -> httpx.Response:
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    last_error: str = ""
    attempt = 0
    for attempt in range(1, attempts + 1):
        try:
            response = client.post(url, json=body)
        except httpx.TransportError as exc:
            last_error = f"network error: {exc}"
        else:
            if response.status_code == 200:
                return response
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in RETRY_STATUS:
                break
        if attempt < attempts:
            wait = 5.0 * 2 ** (attempt - 1)
            log.warning(
                "attempt %d/%d failed (%s); retrying in %.0fs", attempt, attempts, last_error, wait
            )
            sleep(wait)
    raise DownloadError(f"export request failed after {attempt} attempt(s): {last_error}")


def _write_atomic(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def _is_fresh(entry: ManifestEntry | None, path: Path) -> bool:
    return (
        entry is not None
        and entry.final
        and path.exists()
        and path.stat().st_size == entry.bytes
        and sha256_of(path) == entry.sha256
    )


def export_client() -> httpx.Client:
    """HTTP client for the export. It identifies this project and does not pose as the DLD page."""
    return httpx.Client(
        timeout=httpx.Timeout(180.0, connect=30.0), headers={"User-Agent": config.USER_AGENT}
    )


def download_months(
    months: Iterable[Month],
    raw_dir: Path = config.RAW_DIR,
    *,
    today: date | None = None,
    force: bool = False,
    client: httpx.Client | None = None,
    pause_seconds: float = 3.0,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ManifestEntry]:
    """Fetch each month into ``raw_dir`` unless a final, unchanged copy is cached."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    today = today or date.today()
    months = sorted(set(months))
    check_in_window(months, today)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(raw_dir / MANIFEST_NAME)
    own_client = client is None
    client = client or export_client()
    results: list[ManifestEntry] = []
    fetched = 0
    try:
        for month in months:
            name = f"transactions_{month.label}.csv"
            path = raw_dir / name
            cached = manifest.entries.get(name)
            if not force and _is_fresh(cached, path):
                assert cached is not None
                log.info(
                    "cache hit %s (%s bytes, sha256 %s)",
                    name,
                    f"{cached.bytes:,}",
                    cached.sha256[:12],
                )
                results.append(cached)
                continue
            if fetched:
                sleep(pause_seconds)  # be gentle with a public service
            end = min(month.last_day, today)
            response = _post_with_retries(
                client, config.DLD_EXPORT_URL, export_body(month.first_day, end), attempts, sleep
            )
            fetched += 1
            content = response.content
            _, rows = validate_csv(content, response.headers.get("content-type", ""))
            _write_atomic(path, content)
            entry = ManifestEntry(
                file=name,
                source="dld-open-data-export",
                url=config.DLD_EXPORT_URL,
                period_start=month.first_day.isoformat(),
                period_end=end.isoformat(),
                downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                rows=rows,
                final=today >= month.last_day + timedelta(days=PROVISIONAL_DAYS),
            )
            manifest.entries[name] = entry
            manifest.save()
            log.info(
                "downloaded %s: %s rows, %s bytes, sha256 %s%s",
                name,
                f"{rows:,}",
                f"{entry.bytes:,}",
                entry.sha256,
                "" if entry.final else " (provisional: will be refreshed on the next run)",
            )
            results.append(entry)
    finally:
        if own_client:
            client.close()
    return results


def download_dubai_data_sample(
    raw_dir: Path = config.DATA_DIR / "raw" / "dubai_data",
    *,
    client: httpx.Client | None = None,
) -> ManifestEntry:
    """Fetch the public data.dubai sample (about 7,000 rows across all years).

    Useful for checking the schema adapter against the Dubai Pulse column
    names. Too small, and spread over too many years, to train on.
    """
    own_client = client is None
    client = client or httpx.Client(timeout=120.0, headers={"User-Agent": config.USER_AGENT})
    try:
        response = client.get(
            config.DUBAI_DATA_DOWNLOAD_URL, headers={"Accept": "application/json"}
        )
    finally:
        if own_client:
            client.close()
    if response.status_code != 200:
        raise DownloadError(f"data.dubai returned HTTP {response.status_code}")
    payload = response.json()
    rows = payload.get("data")
    if not payload.get("success") or not isinstance(rows, list) or not rows:
        raise DownloadError(f"data.dubai returned no rows: {payload.get('message')!r}")
    header = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = "real_estate_transactions_sample.csv"
    _write_atomic(raw_dir / name, content)
    entry = ManifestEntry(
        file=name,
        source="data.dubai-public-sample",
        url=config.DUBAI_DATA_DOWNLOAD_URL,
        period_start=min(str(r.get("instance_date")) for r in rows),
        period_end=max(str(r.get("instance_date")) for r in rows),
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        rows=len(rows),
        final=True,
    )
    manifest = Manifest(raw_dir / MANIFEST_NAME)
    manifest.entries[name] = entry
    manifest.save()
    log.info(
        "downloaded %s: %s rows, %s bytes, sha256 %s",
        name,
        f"{len(rows):,}",
        f"{entry.bytes:,}",
        entry.sha256,
    )
    return entry
