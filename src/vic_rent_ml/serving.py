"""Artifact-backed serving helpers and standalone FastAPI inference service."""

from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
from fastapi import FastAPI, HTTPException

from .config import FeatureConfig
from .schemas import PredictionRequest, PredictionResponse, UncertaintyBounds
from .recipe import format_postcode, production_feature_frame

_CONFIG = FeatureConfig()
RECIPE_NUM = _CONFIG.numeric_features
RECIPE_CAT = _CONFIG.categorical_features
LEGACY_MIN_HALF_WIDTH = 20.0


def _half_width_for_horizon(residual_bands: dict, horizon: int) -> float:
    widths = residual_bands.get("half_width_p80") or {}
    h = max(1, min(8, horizon if horizon > 0 else 1))
    for key in (h, str(h)):
        if key in widths:
            return float(widths[key])
    if widths:
        return float(max(float(v) for v in widths.values()))
    return LEGACY_MIN_HALF_WIDTH


def calculate_bounds(
    predicted_rent: float,
    *,
    horizon: int,
    residual_bands: Optional[dict] = None,
) -> Tuple[float, float, float]:
    """Calculate (lower, upper, half_width) weekly bounds based on walk-forward calibration."""
    if residual_bands and residual_bands.get("half_width_p80"):
        half_width = _half_width_for_horizon(residual_bands, horizon)
    else:
        half_width = max(predicted_rent * 0.08, LEGACY_MIN_HALF_WIDTH)
    return (
        round(predicted_rent - half_width, 2),
        round(predicted_rent + half_width, 2),
        round(half_width, 2),
    )


def predict_expected_median_from_features(
    model_payload: dict, features: dict
) -> float:
    pipeline = model_payload["model"]
    kind = model_payload.get("kind", "delta")
    lag_1 = float(features.get("_lag_1", 0.0))

    row = dict(features, median_lag_1=lag_1)
    frame = production_feature_frame(pd.DataFrame([row]))
    hat = float(pipeline.predict(frame)[0])
    if kind == "delta":
        return lag_1 + hat
    return hat


def predict_from_artifact(
    payload: dict, postcode: str, apartment_type: str, bedrooms: int,
    year: int, quarter: int, *, suburb_group: str | None = None, region: str | None = None,
) -> dict:
    if payload.get("schema_version") != 2:
        raise ValueError("Retrain with the current CLI to include forecast history")
    anchor = payload["trained_through"]
    origin = anchor["year"] * 4 + anchor["quarter"]
    period = year * 4 + quarter
    horizon = period - origin
    if not 1 <= horizon <= payload["max_horizon"]:
        raise ValueError(f"Forecast must be 1..{payload['max_horizon']} quarters after the artifact's data anchor")
    history = payload["history"]
    eligible = history[
        (history["period"] == origin)
        & (history["apartment_type"] == apartment_type)
        & (history["bedrooms"] == bedrooms)
    ]
    matched = eligible[eligible["postcodes"].map(format_postcode) == format_postcode(postcode)]
    if suburb_group is not None:
        group_match = eligible[eligible["suburb_group"] == suburb_group]
        if not group_match.empty:
            matched = group_match
    imputed = False
    if matched.empty and region is not None:
        matched = eligible[eligible["region"] == region]
        imputed = True
    if matched.empty:
        raise ValueError("No observed history for this location and dwelling; supply a supported suburb group")
    forecasts = payload["forecasts"]
    predictions = forecasts[
        forecasts["suburb_group"].isin(matched["suburb_group"])
        & (forecasts["apartment_type"] == apartment_type)
        & (forecasts["bedrooms"] == bedrooms)
        & (forecasts["period"] == period)
    ]
    if predictions.empty:
        raise ValueError("Artifact has no forecast for this series")
    return {
        "predicted_rent": float(predictions["prediction"].median()), "horizon": horizon,
        "lag_1": float(matched["median"].median()), "lag_4": float(matched["median_lag_4"].median()),
        "suburb_group": str(matched["suburb_group"].iloc[0]) if len(matched) == 1 else region,
        "is_imputed": imputed or len(matched) > 1,
    }


def create_app(model_payload: Optional[dict] = None) -> FastAPI:
    """FastAPI application factory for production containerized inference."""
    app = FastAPI(
        title="vic-rent-ml Inference Service",
        description="Open-source recursive forecasts of Victorian suburb-group moving-annual median rents",
        version="1.0.0",
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "model_loaded": model_payload is not None,
            "recipe": model_payload.get("recipe", "unknown")
            if model_payload
            else None,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(req: PredictionRequest):
        if not model_payload:
            raise HTTPException(
                status_code=503, detail="Model artifact is not loaded."
            )

        anchor = model_payload.get("trained_through")
        if anchor is None:
            raise HTTPException(status_code=503, detail="Artifact lacks history; retrain with the current CLI")
        next_period = anchor["year"] * 4 + anchor["quarter"] + 1
        year = req.year if req.year is not None else (next_period - 1) // 4
        quarter = req.quarter if req.quarter is not None else (next_period - 1) % 4 + 1
        try:
            result = predict_from_artifact(
                model_payload, req.postcode, req.apartment_type, req.bedrooms, year, quarter,
                suburb_group=req.suburb_group,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        pred, horizon = result["predicted_rent"], result["horizon"]
        bands = model_payload.get("residual_bands")
        lower, upper, width = calculate_bounds(
            pred, horizon=horizon, residual_bands=bands
        )

        return PredictionResponse(
            postcode=req.postcode,
            apartment_type=req.apartment_type,
            bedrooms=req.bedrooms,
            year=year,
            quarter=quarter,
            predicted_rent=round(pred, 2),
            bounds=UncertaintyBounds(
                lower_bound=lower,
                upper_bound=upper,
                half_width=width,
                horizon=horizon,
            ),
            recipe=model_payload["recipe"],
            suburb_group=result["suburb_group"],
            is_imputed=result["is_imputed"],
        )

    return app
