"""Turn an API request into a model row, an estimate and its top factors."""

from __future__ import annotations

import difflib
from datetime import date
from typing import Any

import pandas as pd

from dxb_prices import normalise
from dxb_prices.api.schemas import EstimateRequest, EstimateResponse, Factor, ModelInfo, PriceRange
from dxb_prices.features import POI_COLUMNS, month_number
from dxb_prices.model import PriceModel

FEATURE_LABELS: dict[str, str] = {
    "area_sqm": "Size",
    "is_off_plan": "Off-plan",
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
# Beyond this many months after the newest training month, say so.
STALE_MONTHS = 3
_UNSPECIFIED = "__unspecified__"


def factor_value(feature: str, value: str, used_project: str, requested: str | None) -> str:
    """Wording for a factor's value; explains a missing or unrecognised project."""
    if feature == "project" and used_project == _UNSPECIFIED:
        if requested:
            return f"'{requested}' not recognised (treated as a less common project)"
        return "not given (treated as a less common project)"
    return value


class UnknownCommunityError(LookupError):
    def __init__(self, name: str, suggestions: list[str]) -> None:
        super().__init__(f"unknown community {name!r}")
        self.name = name
        self.suggestions = suggestions


class Estimator:
    def __init__(self, model: PriceModel) -> None:
        self.model = model
        self.spec = model.spec
        self.communities = sorted(set(self.spec.community_names.values()))
        self._baseline: dict[str, float] = model.metadata.get("baseline", {}).get("medians", {})
        trained = model.metadata.get("trained_on_months") or [self.spec.reference_month]
        self._first_month = month_number(pd.Series([min(trained)]))[0]
        self._last_month = month_number(pd.Series([max(trained)]))[0]
        self._thin = int(model.metadata.get("thin_community_rows", 50))

    def community_rows(self) -> dict[str, int]:
        return {c: int(self.spec.community_rows.get(c, 0)) for c in self.communities}

    def resolve_community(self, name: str) -> str:
        found = normalise.resolve_name(name, self.spec.community_names)
        if found is None:
            close = difflib.get_close_matches(name, self.communities, n=5, cutoff=0.5)
            lowered = {c.casefold(): c for c in self.communities}
            close += [
                lowered[m]
                for m in difflib.get_close_matches(name.casefold(), list(lowered), n=5, cutoff=0.5)
                if lowered[m] not in close
            ]
            raise UnknownCommunityError(name, close[:5])
        return found

    def _row(
        self, req: EstimateRequest, community: str, warnings: list[str]
    ) -> tuple[pd.DataFrame, str | None]:
        project: str | None = None
        context: dict[str, Any] = self.spec.community_context.get(community, {})
        if req.project:
            project = normalise.resolve_name(req.project, self.spec.project_names)
            if project is None:
                warnings.append(
                    f"Project '{req.project}' is not in the training data; "
                    "the estimate uses community-level information."
                )
            else:
                pctx = self.spec.project_context.get(project, {})
                if pctx.get("community") and pctx["community"] != community:
                    warnings.append(
                        f"Project '{project}' is recorded in {pctx['community']}, not {community}."
                    )
                context = {k: v for k, v in pctx.items() if v is not None} | {
                    k: v for k, v in context.items() if pctx.get(k) is None
                }
        when = req.transaction_date or date.today()
        m = when.year * 12 + when.month - 1
        if m > self._last_month + STALE_MONTHS:
            warnings.append(
                f"The date is {m - self._last_month} months after the newest training "
                "data; the model does not project market movement."
            )
        if m < self._first_month:
            warnings.append("The date is before the training period.")
        freehold = req.freehold if req.freehold is not None else context.get("is_freehold")
        row: dict[str, Any] = {
            "community": community,
            "project": project if project is not None else _UNSPECIFIED,
            "rooms": req.rooms,
            "sub_type": req.sub_type,
            "area_sqm": float(req.size_sqm),
            "is_off_plan": req.off_plan,
            "is_freehold": freehold,
            "month": f"{when.year:04d}-{when.month:02d}",
        }
        for col in POI_COLUMNS:
            row[col] = context.get(col)
        return pd.DataFrame([row]), project

    def estimate(self, req: EstimateRequest) -> EstimateResponse:
        warnings: list[str] = []
        community = self.resolve_community(req.community)
        rows = self.spec.community_rows.get(community, 0)
        if rows < self._thin:
            warnings.append(
                f"Only {rows} training sales in {community}; treat the estimate with extra caution."
            )
        frame, project = self._row(req, community, warnings)
        price = float(self.model.predict(frame)[0])
        low, high = self.model.interval(frame)
        factors = self.model.explain(frame, top_k=5)[0]
        top = []
        used_project = str(frame.loc[0, "project"])
        for f in factors:
            top.append(
                Factor(
                    feature=f.feature,
                    label=FEATURE_LABELS[f.feature],
                    value=factor_value(f.feature, f.value, used_project, req.project),
                    effect_pct=f.effect_pct,
                )
            )
        median = self._baseline.get(community)
        meta = self.model.metadata
        return EstimateResponse(
            estimate_aed=int(round(price, -3)),
            estimate_per_sqm_aed=int(round(price / req.size_sqm, -1)),
            range_80_aed=PriceRange(
                low=int(round(float(low[0]), -3)), high=int(round(float(high[0]), -3))
            ),
            community=community,
            project=project,
            top_factors=top,
            base_per_sqm_aed=int(round(self.model.base_per_sqm(), -1)),
            community_median_per_sqm_aed=int(round(median, -1)) if median else None,
            warnings=warnings,
            model=ModelInfo(
                version=str(meta.get("model_version", "unknown")),
                trained_on_months=list(meta.get("trained_on_months", [])),
                data_period_end=meta.get("data_period_end"),
            ),
        )
