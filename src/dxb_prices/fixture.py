"""Synthetic transactions in the DLD export layout, for tests and demos.

Nothing here comes from DLD data. Community names are public place names
(as spelled in DLD's area list); every price, size, date and transaction
number is generated from the rules below with a fixed seed.

The generator also plants the problems the cleaner must handle: mortgages,
gifts, villas, land and offices, lease-to-own contracts (one listed in both the
Sales and Mortgage groups), exact duplicates, one sale listed twice with
different metro stations, a multi-unit deal, implausible prices, English casing
variants and two Arabic spellings of one name.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dxb_prices.config import FeatureRules, SegmentRules, Settings
from dxb_prices.schema import EXPORT_COLUMNS

SETTINGS = Settings(
    features=FeatureRules(
        min_rows_community=5, min_rows_project=5, min_rows_master_project=5, min_rows_poi=5
    ),
    segments=SegmentRules(thin_community_rows=20),
)

MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04")


@dataclass(frozen=True)
class Community:
    names_en: tuple[str, ...]
    names_ar: tuple[str, ...]
    price_per_sqm: float
    projects: tuple[tuple[str, float], ...]
    metro: str
    mall: str
    landmark: str
    freehold: bool = True
    first_month: int = 0


COMMUNITIES: tuple[Community, ...] = (
    Community(
        ("Marsa Dubai",),
        ("مرسى دبي",),
        22_000,
        (("Harbour View Tower", 1.10), ("Marina Crest", 0.95)),
        "DAMAC Properties",
        "Marina Mall",
        "Burj Al Arab",
    ),
    Community(
        ("BUSINESS BAY", "Business Bay"),
        ("الخليج التجاري", "الخليج التجارى"),
        21_000,
        (("Canal Heights", 1.05), ("Bay Square", 0.92), ("Peninsula One", 1.15)),
        "Business Bay Metro Station",
        "Dubai Mall",
        "Burj Khalifa",
    ),
    Community(
        ("PALM JUMEIRAH", "Palm Jumeirah"),
        ("نخلة جميرا",),
        38_000,
        (("Shoreline Residences", 0.90), ("Crescent Point", 1.20)),
        "",
        "Marina Mall",
        "Burj Al Arab",
    ),
    Community(
        ("Jumeirah Lakes Towers",),
        ("أبراج بحيرات جميرا",),
        15_000,
        (("Lake Terrace", 1.0), ("Cluster Tower", 0.9)),
        "Jumeirah Lakes Towers",
        "Marina Mall",
        "",
    ),
    Community(
        ("Al Barsha South Fourth",),
        ("البرشاء جنوب الرابعة",),
        13_000,
        (("Circle Gardens", 1.0), ("Village Park", 1.08)),
        "",
        "",
        "Sports City Swimming Academy",
    ),
    Community(
        ("DUBAI MARITIME CITY", "Madinat Dubai Almelaheyah"),
        ("مدينة دبي الملاحية",),
        20_000,
        (("Seaside Quay", 1.0),),
        "",
        "",
        "",
    ),
    Community(
        ("Warsan First",),
        ("ورسان الأولى",),
        7_000,
        (("Dragon View", 1.0), ("Lake Court", 0.95)),
        "",
        "Dragon Mart",
        "",
    ),
    Community(
        ("SILICON OASIS",),
        ("واحة دبي للسيليكون",),
        9_000,
        (("Oasis Lofts", 1.0),),
        "",
        "",
        "",
        freehold=False,
    ),
    Community(
        ("Al Jaddaf",),
        ("الجداف",),
        17_000,
        (("Creek Edge", 1.0),),
        "Al Jadaf Metro Station",
        "",
        "Dubai Creek",
        first_month=3,
    ),
)

ROOMS = (("Studio", 38.0), ("1 B/R", 72.0), ("2 B/R", 118.0), ("3 B/R", 175.0))


def _row(**kw: object) -> dict[str, str]:
    row = dict.fromkeys(EXPORT_COLUMNS, "")
    row.update({k: "" if v is None else str(v) for k, v in kw.items()})
    return row


def _planted_problems(
    apartments: list[dict[str, str]], next_id: Callable[[], str]
) -> list[dict[str, str]]:
    """Rows the cleaner must remove or merge, built from generated sales."""
    out: list[dict[str, str]] = []

    def variant(base: dict[str, str], **changes: object) -> dict[str, str]:
        out = dict(base)
        out.update({k: str(v) for k, v in changes.items()})
        out["TRANSACTION_NUMBER"] = next_id()
        return out

    template = apartments[0]
    for i in range(12):
        out.append(
            variant(apartments[i + 10], GROUP_EN="Mortgage", PROCEDURE_EN="Mortgage Registration")
        )
    for i in range(6):
        out.append(variant(apartments[i + 30], GROUP_EN="Gifts", PROCEDURE_EN="Grant"))
    for i in range(4):
        out.append(
            variant(
                apartments[i + 40],
                PROP_TYPE_EN="Building",
                PROP_SB_TYPE_EN="Villa",
                ROOMS_EN="4 B/R",
                ACTUAL_AREA="420",
                PROCEDURE_AREA="420",
            )
        )
        out.append(
            variant(
                apartments[i + 50],
                PROP_TYPE_EN="Land",
                PROP_SB_TYPE_EN="Commercial",
                ROOMS_EN="",
                ACTUAL_AREA="2000",
                PROCEDURE_AREA="2000",
            )
        )
    for i in range(5):
        out.append(variant(apartments[i + 60], PROP_SB_TYPE_EN="Office", ROOMS_EN="Office"))
    for i in range(4):
        out.append(variant(apartments[i + 70], PROCEDURE_EN="Lease to Own Registration"))
    for i in range(5):  # exact duplicates
        out.append(dict(apartments[i + 80]))
    # One sale listed twice, once per nearby metro station.
    out.append({**apartments[90], "NEAREST_METRO_EN": "Another Metro Station"})
    # A lease-to-own contract listed under one number in two groups.
    lease = {
        **apartments[95],
        "TRANSACTION_NUMBER": next_id(),
        "PROCEDURE_EN": "Lease to Own Registration",
    }
    out.extend([lease, {**lease, "GROUP_EN": "Mortgage"}])
    multi_id = next_id()  # one transaction recorded against three units
    for unit_area in ("61.5", "88.0", "120.4"):
        out.append(
            {
                **template,
                "TRANSACTION_NUMBER": multi_id,
                "ACTUAL_AREA": unit_area,
                "PROCEDURE_AREA": unit_area,
                "TRANS_VALUE": "5400000.00",
            }
        )
    out.append(variant(template, TRANS_VALUE="50000.00"))  # below minimum price
    out.append(variant(template, TRANS_VALUE="45000000.00", ACTUAL_AREA="70", PROCEDURE_AREA="70"))
    out.append(variant(template, ACTUAL_AREA="10", PROCEDURE_AREA="10"))  # too small
    out.append(variant(template, TRANS_VALUE="120000.00", ACTUAL_AREA="60", PROCEDURE_AREA="60"))
    out.append(variant(template, TRANS_VALUE=""))  # unparseable price
    out.append(variant(apartments[5], ROOMS_EN="PENTHOUSE"))
    out.append(variant(apartments[6], ROOMS_EN="NA"))
    return out


def generate(seed: int = 7, per_month: int = 140) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    serial = 1

    def next_id() -> str:
        nonlocal serial
        serial += 1
        return f"102-{serial}-2026"

    def stamp(month: str) -> str:
        start = datetime.strptime(month + "-01", "%Y-%m-%d")
        return (
            start
            + timedelta(days=int(rng.integers(0, 28)), seconds=int(rng.integers(28_800, 72_000)))
        ).strftime("%Y-%m-%d %H:%M:%S")

    apartments: list[dict[str, str]] = []
    for m_idx, month in enumerate(MONTHS):
        for _ in range(per_month):
            eligible = [c for c in COMMUNITIES if c.first_month <= m_idx]
            weights = np.array([0.3 if c.first_month else 1.0 for c in eligible])
            comm = eligible[int(rng.choice(len(eligible), p=weights / weights.sum()))]
            project, p_mult = comm.projects[int(rng.integers(0, len(comm.projects)))]
            rooms, base_size = ROOMS[int(rng.choice(4, p=[0.3, 0.35, 0.25, 0.1]))]
            size = round(float(base_size * rng.lognormal(0, 0.12)), 2)
            off_plan = bool(rng.random() < 0.6)
            pps = (
                comm.price_per_sqm
                * p_mult
                * (1.08 if off_plan else 1.0)
                * (1 + 0.005 * m_idx)
                * (base_size / 90) ** -0.08
                * rng.lognormal(0, 0.08)
            )
            price = round(pps * size, -2)
            en = comm.names_en[int(rng.integers(0, len(comm.names_en)))]
            ar = comm.names_ar[int(rng.integers(0, len(comm.names_ar)))]
            if comm.freehold:
                procedure = "Sell - Pre registration" if off_plan else "Sale"
            else:
                procedure = (
                    "Development Registration Pre-Registration" if off_plan else "Sell Development"
                )
            apartments.append(
                _row(
                    TRANSACTION_NUMBER=next_id(),
                    INSTANCE_DATE=stamp(month),
                    GROUP_EN="Sales",
                    PROCEDURE_EN=procedure,
                    IS_OFFPLAN_EN="Off-Plan" if off_plan else "Ready",
                    IS_FREE_HOLD_EN="Free Hold" if comm.freehold else "Non Free Hold",
                    USAGE_EN="Residential",
                    AREA_EN=en,
                    AREA_AR=ar,
                    PROP_TYPE_EN="Unit",
                    PROP_SB_TYPE_EN="Hotel Apartment" if rng.random() < 0.04 else "Flat",
                    TRANS_VALUE=f"{price:.2f}",
                    PROCEDURE_AREA=size,
                    ACTUAL_AREA=size,
                    ROOMS_EN=rooms,
                    PARKING="1",
                    NEAREST_METRO_EN=comm.metro,
                    NEAREST_MALL_EN=comm.mall,
                    NEAREST_LANDMARK_EN=comm.landmark,
                    TOTAL_BUYER="1",
                    TOTAL_SELLER="1",
                    PROJECT_EN=project if rng.random() > 0.05 else "",
                    PROJECT_AR="",
                )
            )
    rows.extend(apartments)

    rows.extend(_planted_problems(apartments, next_id))
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def write(path: Path, seed: int = 7) -> int:
    rows = generate(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(EXPORT_COLUMNS), quoting=csv.QUOTE_ALL, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
