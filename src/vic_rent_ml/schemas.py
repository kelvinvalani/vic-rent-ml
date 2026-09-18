"""Pydantic data validation schemas for prediction requests and responses."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


DwellingType = Literal["flat", "house"]


class PredictionRequest(BaseModel):
    postcode: str = Field(..., pattern=r"^\d{4}$", description="4-digit postcode (e.g. '3000', '3121')")
    apartment_type: DwellingType = Field(..., description="Dwelling classification: 'flat' or 'house'")
    bedrooms: int = Field(..., ge=1, le=5, description="Number of bedrooms (1-5)")
    year: Optional[int] = Field(None, ge=2000, le=2035, description="Target forecast year")
    quarter: Optional[int] = Field(None, ge=1, le=4, description="Target forecast quarter (1-4)")
    suburb_group: Optional[str] = Field(None, description="Official source group, for non-representative postcodes")

    @model_validator(mode="after")
    def paired_date(self):
        if (self.year is None) != (self.quarter is None):
            raise ValueError("Supply both year and quarter, or omit both for the next available quarter")
        return self


class UncertaintyBounds(BaseModel):
    lower_bound: float = Field(..., description="Lower 80% empirical bound in weekly dollars")
    upper_bound: float = Field(..., description="Upper 80% empirical bound in weekly dollars")
    half_width: float = Field(..., description="Empirical half-width in weekly dollars")
    horizon: int = Field(..., description="Forecast horizon in quarters relative to latest anchor")


class PredictionResponse(BaseModel):
    postcode: str
    apartment_type: str
    bedrooms: int
    year: int
    quarter: int
    predicted_rent: float = Field(..., description="Expected moving-annual median weekly rent ($)")
    bounds: UncertaintyBounds
    model_version: str = Field(default="1.0.0")
    recipe: str
    suburb_group: Optional[str] = None
    is_imputed: bool = False


class BenchmarkMetric(BaseModel):
    model_name: str
    val_mae: Optional[float] = None
    test_mae: float
    test_rmse: float
    test_r2: float
    is_winner: bool = False
