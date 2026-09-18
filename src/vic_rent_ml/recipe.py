"""Shared feature contract, recursive forecasting and residual summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FeatureConfig

RECIPE_NUM = FeatureConfig().numeric_features
RECIPE_CAT = FeatureConfig().categorical_features
KEY_COLS = ["suburb_group", "apartment_type", "bedrooms"]


def format_postcode(value) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None", "missing"}:
        return "missing"
    return f"{float(value):.1f}"


def production_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.rename(columns={"distance_to_cbd": "distanceToCbd", "train_dist": "trainDist"}).copy()
    for col in RECIPE_NUM:
        if col not in frame:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["postcodes"] = frame["postcodes"].map(format_postcode)
    return frame[RECIPE_NUM + RECIPE_CAT]


def make_production_pipeline(estimator=None) -> Pipeline:
    est = LinearRegression() if estimator is None else clone(estimator)
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), RECIPE_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RECIPE_CAT),
    ])
    return Pipeline([("pre", pre), ("model", est)])


def fit_production_model(train: pd.DataFrame, kind: str = "delta", estimator=None) -> Pipeline:
    if kind not in {"delta", "level"}:
        raise ValueError("kind must be 'delta' or 'level'")
    target = train["median"] - train["median_lag_1"] if kind == "delta" else train["median"]
    return make_production_pipeline(estimator).fit(production_feature_frame(train), target)


def predict_expected_median(model: Pipeline, panel: pd.DataFrame, kind: str = "delta") -> np.ndarray:
    pred = model.predict(production_feature_frame(panel))
    return pred + panel["median_lag_1"].to_numpy() if kind == "delta" else pred


def forecast_panel(
    model: Pipeline, history: pd.DataFrame, origin: int, horizons: int, kind: str = "delta",
) -> pd.DataFrame:
    """Forecast only from origin-time state, without reading any future rows."""
    history = history[history["period"] <= origin].copy()
    anchors = history[history["period"] == origin].copy()
    if anchors.empty:
        raise ValueError("No observed series at forecast origin")
    state = history[KEY_COLS + ["period", "median"]].copy()
    for lag in (1, 4):
        observed = history[KEY_COLS + ["period", f"median_lag_{lag}"]].rename(
            columns={f"median_lag_{lag}": "median"}
        )
        observed["period"] -= lag
        state = pd.concat([state, observed], ignore_index=True).drop_duplicates(
            KEY_COLS + ["period"], keep="first"
        )
    outputs = []
    for horizon in range(1, horizons + 1):
        period = origin + horizon
        step = anchors.drop(columns=["median", "median_lag_1", "median_lag_4"]).copy()
        for lag in (1, 4):
            previous = state[state["period"] == period - lag][KEY_COLS + ["median"]]
            step = step.merge(previous.rename(columns={"median": f"median_lag_{lag}"}), on=KEY_COLS, how="left")
        fallback = anchors.set_index(KEY_COLS)["median"].reindex(pd.MultiIndex.from_frame(step[KEY_COLS])).to_numpy()
        step["median_lag_4"] = step["median_lag_4"].fillna(pd.Series(fallback, index=step.index))
        step["year"], step["quarter"] = (period - 1) // 4, (period - 1) % 4 + 1
        step["period"] = period
        prediction = predict_expected_median(model, step, kind)
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite model prediction")
        result = step[KEY_COLS + ["period"]].assign(prediction=prediction, origin=origin, horizon=horizon)
        outputs.append(result)
        state = pd.concat([state, result[KEY_COLS + ["period"]].assign(median=prediction)], ignore_index=True)
    return pd.concat(outputs, ignore_index=True)


def walk_forward_abs_errors(
    panel: pd.DataFrame, kind: str = "delta", origins: list[int] | None = None,
    horizons: int = 8, min_train_rows: int = 20, estimator=None,
) -> pd.DataFrame:
    if origins is None:
        origins = list(range(int(panel["period"].min()) + 4, int(panel["period"].max())))
    parts = []
    for origin in origins:
        train = panel[panel["period"] <= origin]
        if len(train) < min_train_rows:
            continue
        model = fit_production_model(train, kind, estimator)
        pred = forecast_panel(model, train, origin, horizons, kind)
        scored = pred.merge(panel[KEY_COLS + ["period", "median"]], on=KEY_COLS + ["period"])
        scored["abs_err"] = (scored["prediction"] - scored["median"]).abs()
        parts.append(scored)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["horizon", "origin", "abs_err"])


def band_table_from_errors(err: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if err.empty:
        raise ValueError("Cannot publish uncertainty bands without calibration errors")
    tbl = err.groupby("horizon")["abs_err"].agg(
        n="size", mae="mean", p50=lambda s: s.quantile(0.5), p80=lambda s: s.quantile(0.8)
    ).reset_index()
    return tbl, {
        "percentile": 80,
        "unit": "weekly_dollars_half_width",
        "half_width_p80": {int(r.horizon): float(r.p80) for r in tbl.itertuples()},
        "mae_by_horizon": {int(r.horizon): float(r.mae) for r in tbl.itertuples()},
        "n_by_horizon": {int(r.horizon): int(r.n) for r in tbl.itertuples()},
        "note": "Retrospective calibration errors; empirical bands, not guaranteed individual-property coverage.",
    }
