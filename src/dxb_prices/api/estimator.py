"""Turn an API request into a model row, an estimate and its top factors."""

from __future__ import annotations

import difflib
from datetime import date
from typing import Any

import pandas as pd

from dxb_prices import normalise, serving
from dxb_prices.api.schemas import (
    EstimateRequest,
    EstimateResponse,
    Factor,
    ModelInfo,
    PriceRange,
    ProjectInfo,
)
from dxb_prices.features import FULL, month_number
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


class UnknownCommunityError(LookupError):
    def __init__(self, name: str, suggestions: list[str]) -> None:
        super().__init__(f"unknown community {name!r}")
        self.name = name
        self.suggestions = suggestions


def _pct(value: Any) -> str | None:
    return f"{float(value) * 100:.1f}%" if isinstance(value, int | float) else None


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
        self._community_model_note = self._accuracy_note()

    def _accuracy_note(self) -> str:
        """How much less accurate the community-level model was on the test month."""
        meta = self.model.metadata
        scores = meta.get("test_scores") or {}
        months = meta.get("evaluated_on_months") or []
        without = _pct((scores.get("model_no_project") or {}).get("mdape"))
        with_project = _pct((scores.get("model") or {}).get("mdape"))
        if not (without and with_project and months):
            return ""
        return (
            f" On the {months[-1]} test month its median error was {without}, "
            f"against {with_project} with the project."
        )

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

    def projects(self, community_name: str) -> list[ProjectInfo]:
        """Projects with training sales in a community, by name, with their training sales."""
        community = self.resolve_community(community_name)
        rows = self.spec.project_rows.get(community, {})
        return [ProjectInfo(name=p, training_sales=int(n)) for p, n in sorted(rows.items())]

    def _project(self, req: EstimateRequest, community: str, warnings: list[str]) -> str | None:
        """The project to use, or None for the community-level model; warns about weak inputs.

        The routing matches ``serving.model_rows``: only a project with training
        sales in this community goes to the full model.
        """
        fallback = (
            ", so the estimate comes from the community-level model, which does not know "
            "the building." + self._community_model_note
        )
        if not req.project:
            warnings.append("No project given" + fallback)
            return None
        project = normalise.resolve_name(req.project, self.spec.project_names)
        if project is None:
            warnings.append(f"Project '{req.project}' is not in the training data" + fallback)
            return None
        if not self.spec.has_project(community, project):
            elsewhere = sorted(c for c, rows in self.spec.project_rows.items() if project in rows)
            warnings.append(
                f"Project '{project}' has no training sales in {community} (it is recorded in "
                f"{', '.join(elsewhere)})" + fallback
            )
            return None
        return project

    def _date(self, req: EstimateRequest, warnings: list[str]) -> date:
        when = req.transaction_date or date.today()
        m = when.year * 12 + when.month - 1
        if m > self._last_month + STALE_MONTHS:
            warnings.append(
                f"The date is {m - self._last_month} months after the newest training "
                "data; the model does not project market movement."
            )
        if m < self._first_month:
            warnings.append("The date is before the training period.")
        return when

    def estimate(self, req: EstimateRequest) -> EstimateResponse:
        warnings: list[str] = []
        community = self.resolve_community(req.community)
        rows = self.spec.community_rows.get(community, 0)
        if rows < self._thin:
            warnings.append(
                f"Only {rows} training sales in {community}; treat the estimate with extra caution."
            )
        project = self._project(req, community, warnings)
        when = self._date(req, warnings)
        request = pd.DataFrame(
            [
                {
                    "community": community,
                    "project": project,
                    "rooms": req.rooms,
                    "sub_type": req.sub_type,
                    "area_sqm": float(req.size_sqm),
                    "is_off_plan": req.off_plan,
                    "is_freehold": req.freehold,
                    "month": f"{when.year:04d}-{when.month:02d}",
                }
            ]
        )
        frame = serving.model_rows(request, self.spec)
        variant = str(frame["variant"].iloc[0])
        price = float(self.model.predict(frame, variant)[0])
        low, high = self.model.interval(frame, variant)
        explanation = self.model.explain(frame, top_k=5, variant=variant)[0]
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
            model_variant="full" if variant == FULL else "community",
            top_factors=[
                Factor(
                    feature=f.feature,
                    label=FEATURE_LABELS[f.feature],
                    value=f.value,
                    effect_pct=f.effect_pct,
                )
                for f in explanation.factors
            ],
            other_factors_effect_pct=explanation.other_effect_pct,
            base_per_sqm_aed=int(round(self.model.base_per_sqm(variant), -1)),
            community_median_per_sqm_aed=int(round(median, -1)) if median else None,
            warnings=warnings,
            model=ModelInfo(
                version=str(meta.get("model_version", "unknown")),
                trained_on_months=list(meta.get("trained_on_months", [])),
                data_period_end=meta.get("data_period_end"),
            ),
        )
