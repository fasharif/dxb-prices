"""Accuracy metrics and the segment breakdown used for error analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Scores:
    rows: int
    mdape: float
    within_10pct: float
    mae_aed: float
    # Median of (estimate - price) / price: below zero means estimates run low.
    median_error: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_float(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def absolute_percentage_errors(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> FloatArray:
    a, p = _as_float(actual), _as_float(predicted)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if np.any(a <= 0):
        raise ValueError("actual prices must be positive")
    return np.abs(p - a) / a


def mdape(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Median absolute percentage error, as a fraction (0.08 means 8%)."""
    return float(np.median(absolute_percentage_errors(actual, predicted)))


def within(actual: npt.ArrayLike, predicted: npt.ArrayLike, tolerance: float = 0.10) -> float:
    """Share of estimates within ``tolerance`` of the real price (inclusive)."""
    return float(np.mean(absolute_percentage_errors(actual, predicted) <= tolerance + 1e-12))


def median_error(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    """Median signed percentage error, as a fraction; shows whether estimates run high or low."""
    a, p = _as_float(actual), _as_float(predicted)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    return float(np.median((p - a) / a))


def mae(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> float:
    a, p = _as_float(actual), _as_float(predicted)
    return float(np.mean(np.abs(p - a)))


def score(actual: npt.ArrayLike, predicted: npt.ArrayLike) -> Scores:
    a = _as_float(actual)
    if a.size == 0:
        raise ValueError("cannot score an empty set")
    return Scores(
        rows=int(a.size),
        mdape=mdape(a, predicted),
        within_10pct=within(a, predicted),
        mae_aed=mae(a, predicted),
        median_error=median_error(a, predicted),
    )


def price_band_labels(prices: pd.Series, bands: tuple[float, ...]) -> pd.Series:
    """Label each price with its band, e.g. "1.0M-2.0M AED" (an ordered categorical)."""

    def fmt(v: float) -> str:
        return f"{v / 1e6:.1f}M"

    labels = []
    for lo, hi in pairwise(bands):
        labels.append(
            f"under {fmt(hi)} AED"
            if lo == 0
            else f"{fmt(lo)} AED and over"
            if np.isinf(hi)
            else f"{fmt(lo)}-{fmt(hi)} AED"
        )
    return pd.cut(prices, bins=list(bands), labels=labels, right=False)


def segment_table(
    frame: pd.DataFrame,
    segment_col: str,
    predictions: dict[str, str],
    actual_col: str = "price_aed",
) -> pd.DataFrame:
    """Scores per segment for each named prediction column.

    ``predictions`` maps a display name ("model", "baseline") to a column. Segments
    come out in category order when ``segment_col`` is categorical.
    """
    rows = []
    for segment, part in frame.groupby(segment_col, observed=True, sort=True):
        for name, col in predictions.items():
            s = score(part[actual_col], part[col])
            rows.append({"segment": str(segment), "estimator": name, **s.as_dict()})
    return pd.DataFrame(rows)
