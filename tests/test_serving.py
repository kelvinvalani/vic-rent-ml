"""Unit tests for standalone serving logic and FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from vic_rent_ml.recipe import fit_production_model, forecast_panel
from vic_rent_ml.serving import (
    calculate_bounds,
    create_app,
)


@pytest.fixture
def dummy_model_payload(sample_panel):
    pipe = fit_production_model(sample_panel, kind="delta")
    history = sample_panel.assign(period=sample_panel.year * 4 + sample_panel.quarter)
    return {
        "model": pipe,
        "kind": "delta",
        "recipe": "linear_fixture",
        "schema_version": 2,
        "trained_through": {"year": 2024, "quarter": 2},
        "history": history,
        "forecasts": forecast_panel(pipe, history, 2024 * 4 + 2, 8),
        "max_horizon": 8,
        "uses_count": False,
        "residual_bands": {
            "percentile": 80,
            "half_width_p80": {"1": 17.23, "2": 30.64, "4": 58.25, "8": 109.36},
        },
    }


def test_calculate_bounds_scales_with_horizon():
    """Verify that uncertainty expands dynamically as forecast horizon increases."""
    bands = {
        "percentile": 80,
        "half_width_p80": {1: 17.23, 2: 30.64, 4: 58.25, 8: 109.36},
    }
    pred = 550.0

    low_1, up_1, width_1 = calculate_bounds(
        pred, horizon=1, residual_bands=bands
    )
    low_4, up_4, width_4 = calculate_bounds(
        pred, horizon=4, residual_bands=bands
    )
    low_8, up_8, width_8 = calculate_bounds(
        pred, horizon=8, residual_bands=bands
    )

    assert width_1 == 17.23
    assert width_4 == 58.25
    assert width_8 == 109.36
    assert width_8 > width_4 > width_1
    assert low_1 < pred < up_1


def test_fastapi_app_serving(dummy_model_payload):
    """Test FastAPI application endpoints."""
    app = create_app(dummy_model_payload)
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"
    assert res.json()["model_loaded"] is True

    # Prediction request
    payload = {
        "postcode": "3000",
        "apartment_type": "flat",
        "bedrooms": 2,
        "year": 2025,
        "quarter": 3,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_rent" in data
    assert data["predicted_rent"] > 0
    assert (
        data["bounds"]["lower_bound"]
        < data["predicted_rent"]
        < data["bounds"]["upper_bound"]
    )
