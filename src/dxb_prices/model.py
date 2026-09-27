"""LightGBM price models: training, persistence, prediction and per-estimate factors.

A saved model holds two boosters that share one feature spec:

* ``full`` uses every feature, including the project and DLD's nearest metro,
  mall and landmark;
* ``community`` leaves those out. It serves requests whose project is not
  given or not in the training data (see ``dxb_prices.serving``).

Both predict the natural log of price per square metre. The price estimate is
``exp(prediction) * size``. Per-estimate factors are TreeSHAP contributions,
which LightGBM computes exactly with ``pred_contrib=True``; all of them
together with the bias term add up to the log prediction, so each factor can
be read as a multiplicative effect on price per square metre.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from dxb_prices import features
from dxb_prices.errors import MissingInputError
from dxb_prices.features import COMMUNITY, FULL, OTHER, VARIANT_FEATURES, FeatureSpec

MODEL_FILES: dict[str, str] = {FULL: "model.lgb", COMMUNITY: "model_community.lgb"}
SPEC_FILE = "features.json"
META_FILE = "metadata.json"

BASE_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
    "min_data_per_group": 50,
    "max_cat_threshold": 64,
    "metric": "l1",
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 42,
}

# A deliberately small search: loss function, tree size and leaf size.
SEARCH_SPACE: tuple[dict[str, Any], ...] = tuple(
    {"objective": objective, "num_leaves": leaves, "min_data_in_leaf": min_leaf}
    | ({"alpha": 0.15} if objective == "huber" else {})
    for objective in ("regression", "huber", "regression_l1")
    for leaves in (31, 127)
    for min_leaf in (20, 60)
)

FloatArray = npt.NDArray[np.float64]


def log_price_per_sqm(frame: pd.DataFrame) -> FloatArray:
    price = np.asarray(frame["price_aed"], dtype=np.float64)
    area = np.asarray(frame["area_sqm"], dtype=np.float64)
    return np.asarray(np.log(price / area), dtype=np.float64)


def train_booster(
    x_train: pd.DataFrame,
    y_train: FloatArray,
    params: dict[str, Any],
    *,
    num_boost_round: int = 5000,
    x_valid: pd.DataFrame | None = None,
    y_valid: FloatArray | None = None,
    early_stopping_rounds: int = 100,
) -> lgb.Booster:
    full = BASE_PARAMS | params
    categorical = [c for c in features.CATEGORICAL_FEATURES if c in x_train.columns]
    dtrain = lgb.Dataset(
        x_train, label=y_train, categorical_feature=categorical, free_raw_data=False
    )
    callbacks: list[Any] = []
    valid_sets: list[lgb.Dataset] = []
    if x_valid is not None and y_valid is not None:
        valid_sets = [lgb.Dataset(x_valid, label=y_valid, reference=dtrain)]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(
        full, dtrain, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks
    )


@dataclass(frozen=True)
class Factor:
    feature: str
    value: str
    effect_pct: float
    contribution: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Explanation:
    """The largest factors of one estimate, and the combined effect of all the others."""

    factors: list[Factor]
    other_contribution: float

    @property
    def other_effect_pct(self) -> float:
        return round((math.exp(self.other_contribution) - 1) * 100, 1)


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _display_value(feature: str, raw: Any, encoded: Any) -> str:
    """Human-readable value of one feature for the factor list."""
    if feature == "area_sqm":
        text = f"{float(raw):,.1f} sqm"
    elif feature in {"is_off_plan", "is_freehold"}:
        text = "unknown" if _is_missing(raw) else ("yes" if bool(raw) else "no")
    elif feature == "month_index":
        text = str(raw)
    elif encoded == OTHER:
        text = f"{raw} (rare or new: grouped as other)"
    elif encoded == features.MISSING:
        text = "not recorded"
    else:
        # The raw value: the model's project level also carries the community.
        text = str(raw)
    return text


class PriceModel:
    def __init__(
        self, boosters: Mapping[str, lgb.Booster], spec: FeatureSpec, metadata: dict[str, Any]
    ) -> None:
        missing = [v for v in VARIANT_FEATURES if v not in boosters]
        if missing:
            raise ValueError(f"a price model needs a booster for each variant; missing {missing}")
        self.boosters = dict(boosters)
        self.spec = spec
        self.metadata = metadata

    # --- prediction ---------------------------------------------------------

    def matrix(self, frame: pd.DataFrame, variant: str = FULL) -> pd.DataFrame:
        return features.transform(frame, self.spec, VARIANT_FEATURES[variant])

    def predict_log_pps(self, frame: pd.DataFrame, variant: str = FULL) -> FloatArray:
        booster = self.boosters[variant]
        return np.asarray(booster.predict(self.matrix(frame, variant)), dtype=np.float64)

    def predict(self, frame: pd.DataFrame, variant: str = FULL) -> FloatArray:
        area = np.asarray(frame["area_sqm"], dtype=np.float64)
        return np.asarray(np.exp(self.predict_log_pps(frame, variant)) * area, dtype=np.float64)

    def interval(self, frame: pd.DataFrame, variant: str = FULL) -> tuple[FloatArray, FloatArray]:
        """80% range from each variant's validation residual quantiles (see metadata)."""
        q = self.metadata["residual_quantiles"][variant]
        log_pps = self.predict_log_pps(frame, variant)
        area = frame["area_sqm"].to_numpy(np.float64)
        return np.exp(log_pps + q["q10"]) * area, np.exp(log_pps + q["q90"]) * area

    # --- explanations ---------------------------------------------------------

    def contributions(self, frame: pd.DataFrame, variant: str = FULL) -> pd.DataFrame:
        """TreeSHAP contributions in log space, one column per feature plus "bias"."""
        x = self.matrix(frame, variant)
        raw = np.asarray(self.boosters[variant].predict(x, pred_contrib=True), dtype=np.float64)
        return pd.DataFrame(raw, columns=[*VARIANT_FEATURES[variant], "bias"], index=frame.index)

    def explain(
        self, frame: pd.DataFrame, top_k: int = 5, variant: str = FULL
    ) -> list[Explanation]:
        x = self.matrix(frame, variant)
        contrib = self.contributions(frame, variant)
        names = list(VARIANT_FEATURES[variant])
        out: list[Explanation] = []
        for idx in frame.index:
            row = contrib.loc[idx, names]
            order = row.abs().sort_values(ascending=False).index
            factors = []
            for feat in order[:top_k]:
                raw_value = frame.loc[idx, "month" if feat == "month_index" else feat]
                value = _display_value(str(feat), raw_value, x.loc[idx, feat])
                c = float(row[feat])
                factors.append(Factor(str(feat), value, round((math.exp(c) - 1) * 100, 1), c))
            out.append(Explanation(factors, float(row[order[top_k:]].sum())))
        return out

    def base_per_sqm(self, variant: str = FULL) -> float:
        """Price per sqm before any feature effect: exp of the TreeSHAP bias term."""
        return float(self.metadata["base_per_sqm"][variant])

    # --- persistence ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for variant, name in MODEL_FILES.items():
            self.boosters[variant].save_model(str(directory / name))
        (directory / SPEC_FILE).write_text(
            json.dumps(self.spec.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (directory / META_FILE).write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> PriceModel:
        needed = [*MODEL_FILES.values(), SPEC_FILE, META_FILE]
        missing = [f for f in needed if not (directory / f).exists()]
        if missing:
            raise MissingInputError(
                f"model directory {directory} is missing {missing}; train one with "
                "`dxb-prices train` (real data) or `dxb-prices train-fixture` (synthetic)"
            )
        boosters = {
            variant: lgb.Booster(model_file=str(directory / name))
            for variant, name in MODEL_FILES.items()
        }
        spec = FeatureSpec.from_dict(
            json.loads((directory / SPEC_FILE).read_text(encoding="utf-8"))
        )
        metadata = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(boosters, spec, metadata)
