"""Unit tests for the feature recipe and uncertainty calibration."""

import pandas as pd
import pytest
from vic_rent_ml.recipe import (
    band_table_from_errors,
    fit_production_model,
    predict_expected_median,
    production_feature_frame,
)


@pytest.fixture
def sample_panel():
    return pd.DataFrame(
        {
            "suburb_group": ["Melbourne", "Melbourne", "Richmond", "Richmond"],
            "apartment_type": ["flat", "flat", "house", "house"],
            "bedrooms": [2, 2, 3, 3],
            "year": [2023, 2024, 2023, 2024],
            "quarter": [1, 2, 1, 2],
            "distance_to_cbd": [2.5, 2.5, 4.0, 4.0],
            "hospitals": [3, 3, 1, 1],
            "schools": [10, 10, 5, 5],
            "train_dist": [0.3, 0.3, 0.8, 0.8],
            "postcodes": ["3000", "3000", "3121", "3121"],
            "median_lag_1": [500.0, 520.0, 700.0, 720.0],
            "median_lag_4": [480.0, 490.0, 680.0, 690.0],
            "median": [520.0, 535.0, 720.0, 750.0],
        }
    )


def test_production_feature_frame_excludes_count(sample_panel):
    """Zero train/serve skew assertion: count must not be in feature frame."""
    features = production_feature_frame(sample_panel)
    assert "count" not in features.columns
    assert "median_lag_4" in features.columns
    assert "distanceToCbd" in features.columns
    assert len(features) == len(sample_panel)


def test_pipeline_fit_and_predict(sample_panel):
    """Verify model fits on delta formulation and reconstructs price levels."""
    pipe = fit_production_model(sample_panel, kind="delta")
    assert pipe is not None

    preds = predict_expected_median(pipe, sample_panel, kind="delta")
    assert len(preds) == len(sample_panel)
    # Rents should be positive realistic dollar values
    assert (preds > 300.0).all()


def test_band_table_from_errors():
    """Verify empirical 80th percentile half-width extraction."""
    err_df = pd.DataFrame(
        {
            "horizon": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "origin": [10, 10, 11, 11, 12, 10, 10, 11, 11, 12],
            "abs_err": [10.0, 12.0, 15.0, 18.0, 20.0, 20.0, 25.0, 30.0, 35.0, 40.0],
        }
    )
    tbl, payload = band_table_from_errors(err_df)
    assert not tbl.empty
    assert 1 in payload["half_width_p80"]
    assert 2 in payload["half_width_p80"]
    # Horizon 2 error should be strictly greater than Horizon 1
    assert payload["half_width_p80"][2] > payload["half_width_p80"][1]
