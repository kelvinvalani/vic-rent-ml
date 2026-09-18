"""Leakage, calendar gaps, artifact identity and application contract regression checks."""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from vic_rent_ml.bakeoff import fit_candidate, recursive_errors
from vic_rent_ml.ingest import attach_lags
from vic_rent_ml.recipe import forecast_panel, production_feature_frame
from vic_rent_ml.serving import create_app, predict_from_artifact
from vic_rent_ml.training import evaluate, train


@pytest.fixture
def panel():
    rows = []
    for group, pc, base in (("Melbourne", "3000", 400), ("Richmond", "3121", 500)):
        for period in range(2015 * 4 + 1, 2025 * 4 + 4):
            rows.append({
                "suburb_group": group, "postcodes": pc, "apartment_type": "flat", "bedrooms": 2,
                "region": "Inner Melbourne", "year": (period - 1) // 4, "quarter": (period - 1) % 4 + 1,
                "median": base + (period - 2015 * 4) * 2 + (period % 4),
            })
    return attach_lags(pd.DataFrame(rows))


def test_calendar_gaps_are_not_previous_observation_lags():
    raw = pd.DataFrame({
        "suburb_group": ["A"] * 6, "apartment_type": ["flat"] * 6, "bedrooms": [2] * 6,
        "year": [2020] * 3 + [2021] * 3, "quarter": [1, 2, 4, 1, 2, 3],
        "median": [100, 200, 400, 500, 600, 700],
    })
    panel = attach_lags(raw)
    assert set(zip(panel.year, panel.quarter)) == {(2021, 1), (2021, 2)}
    assert panel.iloc[0]["median_lag_4"] == 100
    with pytest.raises(ValueError, match="Duplicate"):
        attach_lags(pd.concat([raw, raw.iloc[:1]]))


def test_recursive_forecast_does_not_read_future_actuals(panel):
    origin = 2024 * 4 + 4
    estimator = fit_candidate(panel[panel.period <= origin], "ridge_delta@lags")
    expected = forecast_panel(estimator, panel, origin, 8)
    changed = panel.copy()
    changed.loc[changed.period > origin, ["median", "median_lag_1", "median_lag_4"]] = 1e9
    pd.testing.assert_frame_equal(expected, forecast_panel(estimator, changed, origin, 8))


def test_recursive_persistence_and_seasonal_baselines(panel):
    origin = 2024 * 4 + 4
    for name in ("naive_persist", "naive_seasonal"):
        errors = recursive_errors(panel, name, [origin], 8, origin + 1, origin + 8)
        assert len(errors) == 6
        if name == "naive_persist":
            assert (errors.groupby("suburb_group")["prediction"].nunique() == 1).all()


def test_selected_tree_is_the_refitted_served_tree(panel, tmp_path):
    name = "xgb_d3_delta@lags"
    payload = train(panel, tmp_path, candidate_names=[name])
    # This increasing panel must favor a learned delta over a flat forecast.
    assert payload["recipe"] == name
    assert isinstance(payload["model"].named_steps["model"], XGBRegressor)
    restored = joblib.load(tmp_path / "production_clone.joblib")
    anchor = 2025 * 4 + 3
    direct = forecast_panel(restored["model"], restored["history"], anchor, 8)
    pd.testing.assert_frame_equal(direct, restored["forecasts"])
    client = TestClient(create_app(restored))
    for horizon in (1, 4, 8):
        year, q = (anchor + horizon - 1) // 4, (anchor + horizon - 1) % 4 + 1
        result = predict_from_artifact(restored, "3000", "flat", 2, year, q)
        actual = client.post("/predict", json={
            "postcode": "3000", "apartment_type": "flat", "bedrooms": 2, "year": year, "quarter": q,
        })
        assert actual.status_code == 200
        expected = direct[(direct.suburb_group == "Melbourne") & (direct.horizon == horizon)].prediction.iloc[0]
        assert result["predicted_rent"] == pytest.approx(expected)
        assert actual.json()["predicted_rent"] == round(expected, 2)
    assert client.post("/predict", json={
        "postcode": "9999", "apartment_type": "house", "bedrooms": 5,
    }).status_code == 422
    with pytest.raises(ValueError, match="1..8"):
        predict_from_artifact(restored, "3000", "flat", 2, 2030, 1)


def test_selection_reuse_rejects_changed_validation_data(panel, tmp_path):
    evaluate(panel, tmp_path, candidate_names=["ridge_delta@lags"])
    changed = panel.copy()
    changed.loc[changed.year == 2023, "median"] += 10
    with pytest.raises(ValueError, match="Selection data changed"):
        train(changed, tmp_path, reuse_selection=True)


def test_no_random_split_fallback(panel, tmp_path):
    with pytest.raises(ValueError, match="later test"):
        evaluate(panel[panel.year < 2025], tmp_path, candidate_names=[])


def test_forecast_delta_uses_observed_then_predicted_seasonal_lags(panel):
    origin = 2024 * 4 + 4
    estimator = Pipeline([("model", DummyRegressor(strategy="constant", constant=5))])
    estimator.fit(production_feature_frame(panel), np.zeros(len(panel)))
    predictions = forecast_panel(estimator, panel, origin, 8)
    base = panel[(panel.period == origin) & (panel.suburb_group == "Melbourne")]["median"].iloc[0]
    assert list(predictions[predictions.suburb_group == "Melbourne"]["prediction"]) == [
        base + 5 * horizon for horizon in range(1, 9)
    ]
