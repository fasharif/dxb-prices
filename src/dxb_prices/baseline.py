"""Baselines: simple lookups of median price per square metre, times the size.

* ``CommunityMedianBaseline``: the community's median. This is the baseline
  the brief asks for.
* ``ProjectMedianBaseline``: the project's own median when it has enough
  training sales, otherwise the community's. Most sales are off-plan units
  priced from a developer's list, so knowing the building is most of what a
  model can learn; this baseline shows how much the model adds beyond that.

The medians come only from the rows a baseline is fitted on (the training
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


@dataclass
class ProjectMedianBaseline:
    """Median price per sqm of the (community, project) pair, else of the community."""

    medians: dict[str, dict[str, float]]
    community: CommunityMedianBaseline
    min_rows: int

    @classmethod
    def fit(cls, train: pd.DataFrame, min_rows: int) -> ProjectMedianBaseline:
        community = CommunityMedianBaseline.fit(train)
        pps = train["price_aed"] / train["area_sqm"]
        keyed = pd.DataFrame(
            {"community": train["community"], "project": train["project"], "pps": pps}
        ).dropna(subset=["community", "project"])
        stats = keyed.groupby(["community", "project"])["pps"].agg(["median", "size"])
        kept = stats[stats["size"] >= min_rows].reset_index()
        medians: dict[str, dict[str, float]] = {}
        for comm, proj, median in zip(
            kept["community"], kept["project"], kept["median"], strict=True
        ):
            medians.setdefault(str(comm), {})[str(proj)] = float(median)
        return cls(medians=medians, community=community, min_rows=min_rows)

    def per_sqm(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        fallback = self.community.per_sqm(frame)
        pairs = zip(
            frame["community"].astype("object"), frame["project"].astype("object"), strict=True
        )
        own = [
            self.medians.get(c, {}).get(p) if isinstance(c, str) and isinstance(p, str) else None
            for c, p in pairs
        ]
        return np.asarray(
            [f if o is None else o for o, f in zip(own, fallback, strict=True)], dtype=np.float64
        )

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        area = np.asarray(frame["area_sqm"], dtype=np.float64)
        return np.asarray(self.per_sqm(frame) * area, dtype=np.float64)
