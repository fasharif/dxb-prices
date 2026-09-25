"""Request and response models for the estimate API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dxb_prices.config import DEFAULT_SETTINGS

RoomsLabel = Literal["studio", "1", "2", "3", "4", "5+", "penthouse"]
_RULES = DEFAULT_SETTINGS.cleaning


class EstimateRequest(BaseModel):
    """One apartment to value."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "community": "Business Bay",
                    "size_sqm": 75,
                    "rooms": "1",
                    "off_plan": False,
                    "project": None,
                    "transaction_date": None,
                }
            ]
        },
    )

    community: str = Field(
        min_length=2,
        max_length=120,
        description="DLD area name in English or Arabic, e.g. 'Marsa Dubai'",
    )
    size_sqm: float = Field(
        ge=_RULES.min_area_sqm, le=_RULES.max_area_sqm, description="Internal area in square metres"
    )
    rooms: RoomsLabel = Field(description="studio, 1-4, 5+ or penthouse")
    off_plan: bool = Field(description="true for an off-plan sale, false for a ready unit")
    project: str | None = Field(
        default=None, max_length=160, description="Project or building name, if known"
    )
    sub_type: Literal["Flat", "Hotel Apartment"] = "Flat"
    freehold: bool | None = Field(
        default=None, description="Leave empty to use the usual status for the area"
    )
    transaction_date: date | None = Field(
        default=None, description="Date to value at; defaults to today"
    )

    @field_validator("rooms", mode="before")
    @classmethod
    def _rooms(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            if value < 0:
                return value
            return "studio" if value == 0 else "5+" if value >= 5 else str(value)
        if isinstance(value, str):
            return value.strip().casefold()
        return value

    @field_validator("community", "project", mode="after")
    @classmethod
    def _strip(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class Factor(BaseModel):
    feature: str
    label: str
    value: str
    effect_pct: float = Field(description="Effect on price per sqm, in percent")


class PriceRange(BaseModel):
    low: int
    high: int


class ModelInfo(BaseModel):
    version: str
    trained_on_months: list[str]
    data_period_end: str | None


class EstimateResponse(BaseModel):
    estimate_aed: int
    estimate_per_sqm_aed: int
    range_80_aed: PriceRange
    community: str
    project: str | None
    top_factors: list[Factor]
    base_per_sqm_aed: int = Field(description="Starting point before any factor is applied")
    community_median_per_sqm_aed: int | None = Field(
        description="Median price per sqm of this community's training sales (the baseline)"
    )
    warnings: list[str]
    model: ModelInfo


class Problem(BaseModel):
    detail: str
    suggestions: list[str] = []
