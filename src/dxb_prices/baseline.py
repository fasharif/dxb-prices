"""Baseline: the community's median price per square metre, times the size.

The medians come only from the rows the baseline is fitted on (the training
period). A community with no training sales falls back to the median of all
training sales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class CommunityMedianBaseline:
    medians: dict[str, float]
    global_median: float

    @classmethod
    def fit(cls, train: pd.DataFrame) -> CommunityMedianBaseline:
        if train.empty:
            raise ValueError("cannot fit the baseline on an empty training set")
        pps = train["price_aed"] / train["area_sqm"]
        medians = pps.groupby(train["community"]).median()
        return cls(
            medians={str(k): float(v) for k, v in medians.items()},
            global_median=float(pps.median()),
        )

    def per_sqm(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        mapped = frame["community"].astype("object").map(self.medians)
        return np.asarray(mapped.fillna(self.global_median), dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        area = np.asarray(frame["area_sqm"], dtype=np.float64)
        return np.asarray(self.per_sqm(frame) * area, dtype=np.float64)

    def covered(self, frame: pd.DataFrame) -> pd.Series:
        """True where the community had training sales (no global fallback)."""
        return frame["community"].astype("object").isin(self.medians.keys())

    def to_dict(self) -> dict[str, Any]:
        return {"medians": self.medians, "global_median": self.global_median}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommunityMedianBaseline:
        return cls(
            medians={str(k): float(v) for k, v in data["medians"].items()},
            global_median=float(data["global_median"]),
        )
