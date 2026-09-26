"""Paths, source URLs and the numeric rules the pipeline applies.

Every threshold that changes which rows the model sees lives here, so that
docs/data.md and the code cannot drift apart silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Locations -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


DATA_DIR = _env_path("DXB_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw" / "dld"
ARTIFACTS_DIR = _env_path("DXB_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MODEL_DIR = _env_path("DXB_MODEL_DIR", ARTIFACTS_DIR / "model")
REPORTS_DIR = _env_path("DXB_REPORTS_DIR", PROJECT_ROOT / "reports")

# --- Sources -------------------------------------------------------------------

# The CSV export behind the "Download as CSV" button on the DLD open data page
# (https://dubailand.gov.ae/en/open-data/real-estate-data/). The request body
# mirrors what the page itself sends.
DLD_EXPORT_URL = "https://gateway.dubailand.gov.ae/open-data/transactions/export/csv"
DLD_PAGE_URL = "https://dubailand.gov.ae/en/open-data/real-estate-data/"

# Dubai Pulse (dubaipulse.gov.ae) now redirects to data.dubai. Its public
# download for "Real Estate Transactions" is a 7,000-row sample across all
# years; the full table needs an approved API key.
DUBAI_DATA_DATASET_ID = 470061
DUBAI_DATA_DOWNLOAD_URL = (
    "https://data.dubai/o/dda/data-services/dataset-metadata"
    f"?datasetId={DUBAI_DATA_DATASET_ID}&download=true"
)
DUBAI_DATA_PAGE_URL = f"https://data.dubai/en/l/{DUBAI_DATA_DATASET_ID}"

USER_AGENT = "dxb-prices/0.1 (personal portfolio project; +https://github.com/fasharif/dxb-prices)"

# --- Cleaning rules ------------------------------------------------------------


@dataclass(frozen=True)
class CleaningRules:
    """Fixed validity rules applied to every row, in every split.

    They remove records that cannot be a real arm's-length apartment sale.
    They are deliberately loose: they only use the price where a value is
    physically implausible, so evaluation rows are not filtered towards the
    easy cases.
    """

    # The largest genuine penthouses in the 2026 data are just under 3,000 sqm.
    min_area_sqm: float = 18.0
    max_area_sqm: float = 3_000.0
    min_price_aed: float = 100_000.0
    max_price_aed: float = 500_000_000.0
    # About 230 AED per sq ft: below any open-market apartment price in Dubai
    # in 2026. Rows under it are part-share transfers or keying errors.
    min_price_per_sqm: float = 2_500.0
    # Branded penthouses reach about 184,000 AED/sqm; values above 250,000
    # in this data are keying errors or multi-unit deals booked on one unit.
    max_price_per_sqm: float = 250_000.0
    # Sub-types that describe a residential unit someone lives in.
    residential_sub_types: tuple[str, ...] = ("Flat", "Hotel Apartment")
    # Procedures inside the "Sales" group that are sales of a unit. The
    # "Development" names are the same sales in non-freehold development zones
    # (Dubai Investment Park, Silicon Oasis): 98% of those apartment sales are flagged
    # "Non Free Hold". Lease-to-own contracts are excluded because their
    # recorded value is not a purchase price.
    market_sale_procedures: tuple[str, ...] = (
        "Sale",
        "Sell",
        "Sell - Pre registration",
        "Delayed Sell",
        "Sale On Payment Plan",
        "Sell Development",
        "Sell Development - Pre Registration",
        "Development Registration",
        "Development Registration Pre-Registration",
        "Delayed Development",
        "Delayed Sell Development",
    )


@dataclass(frozen=True)
class TrainingTrim:
    """Extra trimming applied only to the rows the model learns from.

    Percentiles are computed on the training rows alone, per community where
    possible, so no information from validation or test months leaks in.
    """

    lower_quantile: float = 0.005
    upper_quantile: float = 0.995


@dataclass(frozen=True)
class FeatureRules:
    """Controls for high-cardinality categorical fields."""

    # A category needs this many training rows to keep its own level;
    # anything rarer (or unseen) is mapped to OTHER.
    min_rows_community: int = 30
    min_rows_project: int = 40
    min_rows_poi: int = 30


@dataclass(frozen=True)
class BaselineRules:
    """The project-median baseline uses a project's own median only with this many sales."""

    min_project_rows: int = 5


@dataclass(frozen=True)
class SplitRules:
    """Temporal split: the newest months are held out."""

    test_months: int = 1
    valid_months: int = 1
    min_train_months: int = 2


@dataclass(frozen=True)
class SegmentRules:
    """Boundaries used for the error analysis."""

    # AED; bands are [lower, upper).
    price_bands: tuple[float, ...] = (0.0, 1_000_000.0, 2_000_000.0, 5_000_000.0, float("inf"))
    # Training rows a community needs to count as established.
    thin_community_rows: int = 50


@dataclass(frozen=True)
class Settings:
    cleaning: CleaningRules = field(default_factory=CleaningRules)
    trim: TrainingTrim = field(default_factory=TrainingTrim)
    features: FeatureRules = field(default_factory=FeatureRules)
    split: SplitRules = field(default_factory=SplitRules)
    segments: SegmentRules = field(default_factory=SegmentRules)
    baseline: BaselineRules = field(default_factory=BaselineRules)
    random_seed: int = 42


DEFAULT_SETTINGS = Settings()
