"""Global SHAP summary for a trained model (training-time only; needs the ``train`` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dxb_prices.features import FULL, VARIANT_FEATURES
from dxb_prices.model import PriceModel

FEATURE_LABELS: dict[str, str] = {
    "area_sqm": "Size (sqm)",
    "is_off_plan": "Off-plan or ready",
    "is_freehold": "Freehold",
    "month_index": "Month of sale",
    "community": "Community",
    "project": "Project",
    "rooms": "Rooms",
    "sub_type": "Flat or hotel apartment",
    "nearest_metro": "Nearest metro",
    "nearest_mall": "Nearest mall",
    "nearest_landmark": "Nearest landmark",
}

# One series colour on a light surface; values are printed at the bar ends.
_SURFACE = "#fcfcfb"
_SERIES = "#2a78d6"
_TEXT = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_AXIS = "#d9d8d4"


def shap_values(
    model: PriceModel, frame: pd.DataFrame, variant: str = FULL
) -> tuple[np.ndarray, float]:
    """SHAP values from the ``shap`` library, and their largest gap to LightGBM's own values.

    Serving uses LightGBM's ``pred_contrib`` so the API image does not need
    ``shap``; the returned gap shows the two agree.
    """
    import shap

    x = model.matrix(frame, variant)
    explainer = shap.TreeExplainer(model.boosters[variant])
    values = np.asarray(explainer.shap_values(x), dtype=np.float64)
    native = model.contributions(frame, variant)[list(VARIANT_FEATURES[variant])].to_numpy()
    return values, float(np.max(np.abs(values - native)))


def global_importance(
    model: PriceModel,
    frame: pd.DataFrame,
    sample: int = 5000,
    seed: int = 42,
    variant: str = FULL,
) -> tuple[pd.DataFrame, float]:
    rows = (
        frame.sample(n=min(sample, len(frame)), random_state=seed) if len(frame) > sample else frame
    )
    values, gap = shap_values(model, rows, variant)
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame({"feature": list(VARIANT_FEATURES[variant]), "mean_abs_shap": mean_abs})
    table["label"] = table["feature"].map(FEATURE_LABELS)
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), gap


def plot_importance(table: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = table.sort_values("mean_abs_shap", ascending=True)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * n + 1.3), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.barh(ordered["label"], ordered["mean_abs_shap"], height=0.55, color=_SERIES)
    top = float(ordered["mean_abs_shap"].max())
    for y, value in enumerate(ordered["mean_abs_shap"]):
        ax.text(
            value + top * 0.015,
            y,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            color=_TEXT_SECONDARY,
        )
    ax.set_xlim(0, top * 1.15)
    ax.set_xlabel("Mean |SHAP value| on log price per sqm", fontsize=8.5, color=_TEXT_SECONDARY)
    ax.set_title(title, fontsize=10, color=_TEXT, loc="left", pad=10)
    ax.tick_params(axis="y", labelsize=8.5, colors=_TEXT, length=0)
    ax.tick_params(axis="x", labelsize=8, colors=_TEXT_SECONDARY, color=_AXIS)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
