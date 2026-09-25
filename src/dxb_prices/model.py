"""LightGBM price model: training, persistence, prediction and per-estimate factors.

The model predicts the natural log of price per square metre. The price
estimate is ``exp(prediction) * size``. Per-estimate factors are TreeSHAP
contributions, which LightGBM computes exactly with ``pred_contrib=True``;
they add up (with the bias term) to the log prediction, so each factor can be
read as a multiplicative effect on price per square metre.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

from dxb_prices import features
from dxb_prices.features import OTHER, FeatureSpec

MODEL_FILE = "model.lgb"
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
    dtrain = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=list(features.CATEGORICAL_FEATURES),
        free_raw_data=False,
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
        text = str(encoded)
    return text


class PriceModel:
    def __init__(self, booster: lgb.Booster, spec: FeatureSpec, metadata: dict[str, Any]) -> None:
        self.booster = booster
        self.spec = spec
        self.metadata = metadata

    # --- prediction ---------------------------------------------------------

    def matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return features.transform(frame, self.spec)

    def predict_log_pps(self, frame: pd.DataFrame) -> FloatArray:
        return np.asarray(self.booster.predict(self.matrix(frame)), dtype=np.float64)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        area = np.asarray(frame["area_sqm"], dtype=np.float64)
        return np.asarray(np.exp(self.predict_log_pps(frame)) * area, dtype=np.float64)

    def interval(self, frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
        """80% range from validation residual quantiles (see metadata["residual_quantiles"])."""
        q = self.metadata["residual_quantiles"]
        log_pps = self.predict_log_pps(frame)
        area = frame["area_sqm"].to_numpy(np.float64)
        return np.exp(log_pps + q["q10"]) * area, np.exp(log_pps + q["q90"]) * area

    # --- explanations ---------------------------------------------------------

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log space, one column per feature plus "bias"."""
        x = self.matrix(frame)
        raw = np.asarray(self.booster.predict(x, pred_contrib=True), dtype=np.float64)
        return pd.DataFrame(raw, columns=[*features.FEATURES, "bias"], index=frame.index)

    def explain(self, frame: pd.DataFrame, top_k: int = 5) -> list[list[Factor]]:
        x = self.matrix(frame)
        contrib = self.contributions(frame)
        out: list[list[Factor]] = []
        for idx in frame.index:
            row = contrib.loc[idx, list(features.FEATURES)]
            order = row.abs().sort_values(ascending=False).index[:top_k]
            factors = []
            for feat in order:
                raw_value = frame.loc[idx, "month" if feat == "month_index" else feat]
                value = _display_value(str(feat), raw_value, x.loc[idx, feat])
                c = float(row[feat])
                factors.append(Factor(str(feat), value, round((math.exp(c) - 1) * 100, 1), c))
            out.append(factors)
        return out

    def base_per_sqm(self) -> float:
        """Price per sqm before any feature effect: exp of the TreeSHAP bias term."""
        return float(self.metadata["base_per_sqm"])

    # --- persistence ----------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / MODEL_FILE))
        (directory / SPEC_FILE).write_text(
            json.dumps(self.spec.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (directory / META_FILE).write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> PriceModel:
        missing = [f for f in (MODEL_FILE, SPEC_FILE, META_FILE) if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"model directory {directory} is missing {missing}; train one with "
                "`dxb-prices train` (real data) or `dxb-prices train-fixture` (synthetic)"
            )
        booster = lgb.Booster(model_file=str(directory / MODEL_FILE))
        spec = FeatureSpec.from_dict(
            json.loads((directory / SPEC_FILE).read_text(encoding="utf-8"))
        )
        metadata = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return cls(booster, spec, metadata)
