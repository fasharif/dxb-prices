"""Temporal train / validation / test split and training-only trimming."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dxb_prices.config import SplitRules, TrainingTrim


class InsufficientDataError(ValueError):
    """Not enough distinct months for a temporal split."""


@dataclass(frozen=True)
class TemporalSplit:
    train_months: tuple[str, ...]
    valid_months: tuple[str, ...]
    test_months: tuple[str, ...]

    def describe(self) -> str:
        def span(ms: tuple[str, ...]) -> str:
            return ms[0] if len(ms) == 1 else f"{ms[0]} to {ms[-1]}"

        return (
            f"train {span(self.train_months)}, validate {span(self.valid_months)}, "
            f"test {span(self.test_months)}"
        )


def plan(months: Iterable[str], rules: SplitRules) -> TemporalSplit:
    """Oldest months train, then validation, then the newest months test."""
    ordered = sorted(set(months))
    needed = rules.min_train_months + rules.valid_months + rules.test_months
    if len(ordered) < needed:
        raise InsufficientDataError(
            f"need at least {needed} months of data for a temporal split "
            f"({rules.min_train_months} train, {rules.valid_months} validation, "
            f"{rules.test_months} test); found {len(ordered)}: {ordered}"
        )
    n_test, n_valid = rules.test_months, rules.valid_months
    return TemporalSplit(
        train_months=tuple(ordered[: -(n_test + n_valid)]),
        valid_months=tuple(ordered[-(n_test + n_valid) : -n_test]),
        test_months=tuple(ordered[-n_test:]),
    )


def apply(
    frame: pd.DataFrame, split: TemporalSplit
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["month"].isin(split.train_months)].copy()
    valid = frame.loc[frame["month"].isin(split.valid_months)].copy()
    test = frame.loc[frame["month"].isin(split.test_months)].copy()
    return train, valid, test


def trim_training(
    train: pd.DataFrame, rules: TrainingTrim, min_group_rows: int = 200
) -> pd.DataFrame:
    """Drop the extreme price-per-sqm tails from training rows only.

    Communities with at least ``min_group_rows`` rows lose the tails of their
    own distribution. Smaller communities are not trimmed: a few rows give no
    reliable percentiles, and the percentiles of the whole training set would
    remove entire luxury communities (in the 2026 data, every training sale in
    Jumeirah Second is above the overall 99.5th percentile). The fixed validity
    rules in ``clean`` still apply to them. Nothing here looks at validation or
    test rows.
    """
    ratio = (train["price_aed"] / train["area_sqm"]).to_numpy(np.float64)
    log_pps = pd.Series(np.log(ratio), index=train.index)
    counts = train["community"].map(train["community"].value_counts())
    grouped = log_pps.groupby(train["community"])
    lo = grouped.transform(lambda s: s.quantile(rules.lower_quantile))
    hi = grouped.transform(lambda s: s.quantile(rules.upper_quantile))
    big = counts >= min_group_rows
    keep = ~big | ((log_pps >= lo) & (log_pps <= hi))
    kept: pd.DataFrame = train.loc[keep].copy()
    return kept
