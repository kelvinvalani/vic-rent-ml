"""Family-neutral, recursive temporal model selection."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .recipe import forecast_panel, production_feature_frame


@dataclass(frozen=True)
class EvaluationPlan:
    validation_origins: tuple[int, ...] = tuple(year * 4 + 4 for year in (2018, 2019, 2020, 2021))
    validation_end: int = 2023 * 4 + 4
    calibration_start: int = 2024 * 4 + 1
    test_origin: int = 2024 * 4 + 4
    horizons: int = 8


def build_model_zoo() -> dict[str, tuple[str, Pipeline]]:
    """Every family sees the same features, origins, targets and preprocessing."""
    estimators = {
        "lin": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=0.1, max_iter=5000),
        "enet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "huber": HuberRegressor(max_iter=2000),
        "bayes": BayesianRidge(),
        "hgb": HistGradientBoostingRegressor(
            max_iter=200, max_leaf_nodes=15, l2_regularization=10, early_stopping=False, random_state=42
        ),
        "rf": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=10, n_jobs=2, random_state=42
        ),
        "et": ExtraTreesRegressor(
            n_estimators=150, max_depth=12, min_samples_leaf=10, n_jobs=2, random_state=42
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=10, random_state=42
        ),
    }
    for depth in (3, 6):
        estimators[f"xgb_d{depth}"] = XGBRegressor(
            n_estimators=300, max_depth=depth, learning_rate=0.05,
            min_child_weight=10, reg_lambda=10, subsample=0.8, colsample_bytree=0.8,
            n_jobs=2, random_state=42, tree_method="hist", objective="reg:squarederror",
        )
    zoo = {}
    for features in ("lags", "location"):
        num = ["quarter", "year", "bedrooms", "median_lag_1", "median_lag_4"]
        cat = ["apartment_type"] + (["postcodes"] if features == "location" else [])
        for name, estimator in estimators.items():
            pre = ColumnTransformer([
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), num),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ])
            zoo[f"{name}_delta@{features}"] = ("delta", Pipeline([
                ("pre", pre), ("model", clone(estimator)),
            ]))
    return zoo


def choose_winner(metrics_df: pd.DataFrame) -> dict:
    """Select on validation only; exact ties use RMSE then a stable candidate ID."""
    scored = metrics_df.copy()
    if scored.empty or "val_mae" not in scored:
        raise ValueError("Validation scores are required for model selection")
    scored = scored[np.isfinite(scored["val_mae"]) & np.isfinite(scored["val_rmse"])]
    if "error" in scored:
        scored = scored[scored["error"].isna()]
    if scored.empty:
        raise ValueError("No candidate has finite validation scores")
    chosen = scored.sort_values(["val_mae", "val_rmse", "model"], kind="stable").iloc[0]
    return {
        "status": "winner_selected",
        "model": str(chosen["model"]),
        "kind": str(chosen["kind"]),
        "val_mae": float(chosen["val_mae"]),
        "val_rmse": float(chosen["val_rmse"]),
        "selection_metric": "equal-horizon mean recursive validation MAE (AUD/week)",
    }


def fit_candidate(panel: pd.DataFrame, name: str) -> Pipeline:
    if name == "naive_persist":
        estimator = DummyRegressor(strategy="constant", constant=0.0)
        return Pipeline([("model", estimator)]).fit(
            production_feature_frame(panel), panel["median"] - panel["median_lag_1"]
        )
    if name == "naive_seasonal":
        return Pipeline([
            ("pre", ColumnTransformer([("lag4", "passthrough", ["median_lag_4"])])),
            ("model", LinearRegression(fit_intercept=False)),
        ]).fit(production_feature_frame(panel), panel["median_lag_4"])
    zoo = build_model_zoo()
    if name not in zoo:
        raise ValueError(f"Unknown candidate: {name}")
    _, template = zoo[name]
    return clone(template).fit(production_feature_frame(panel), panel["median"] - panel["median_lag_1"])


def recursive_errors(
    panel: pd.DataFrame, name: str, origins: tuple[int, ...] | list[int],
    horizons: int, target_min: int, target_max: int,
) -> pd.DataFrame:
    chunks = []
    kind = "level" if name == "naive_seasonal" else "delta"
    for origin in origins:
        train = panel[panel["period"] <= origin]
        if train.empty or not (train["period"] == origin).any():
            raise ValueError(f"No training data at origin {origin}")
        model = fit_candidate(train, name)
        predictions = forecast_panel(model, train, origin, horizons, kind=kind)
        actual = panel[["suburb_group", "apartment_type", "bedrooms", "period", "median"]]
        scored = predictions.merge(
            actual, on=["suburb_group", "apartment_type", "bedrooms", "period"], validate="one_to_one"
        )
        scored = scored[scored["period"].between(target_min, target_max)].copy()
        scored["error"] = scored["prediction"] - scored["median"]
        scored["abs_err"] = scored["error"].abs()
        chunks.append(scored)
    if not chunks:
        raise ValueError("At least one forecast origin is required")
    return pd.concat(chunks, ignore_index=True)


def error_metrics(errors: pd.DataFrame, prefix: str) -> dict:
    if errors.empty or not np.isfinite(errors["prediction"]).all():
        raise ValueError("Empty or non-finite forecast evaluation")
    mae = errors.groupby("horizon")["abs_err"].mean().mean()
    rmse = errors.assign(squared=errors["error"] ** 2).groupby("horizon")["squared"].mean().pow(0.5).mean()
    return {f"{prefix}_mae": float(mae), f"{prefix}_rmse": float(rmse), f"{prefix}_n": len(errors)}


def validate_plan(panel: pd.DataFrame, plan: EvaluationPlan) -> None:
    if not plan.validation_origins or not 1 <= plan.horizons <= 8:
        raise ValueError("Specify validation origins and 1..8 horizons")
    if not max(plan.validation_origins) < plan.validation_end < plan.calibration_start <= plan.test_origin:
        raise ValueError("Validation, calibration and test must be chronologically ordered")
    if panel["period"].max() <= plan.test_origin:
        raise ValueError("A later test window is required; no random-split fallback")
    if panel.duplicated(["suburb_group", "apartment_type", "bedrooms", "period"]).any():
        raise ValueError("Duplicate panel keys")
    if not np.isfinite(panel[["median", "median_lag_1", "median_lag_4"]].to_numpy()).all():
        raise ValueError("Targets and required lags must be finite")


def run_bakeoff(
    panel: pd.DataFrame, *, out_dir: Path | None = None,
    plan: EvaluationPlan | None = None, candidate_names: list[str] | None = None,
) -> tuple[dict, pd.DataFrame]:
    plan = plan or EvaluationPlan()
    validate_plan(panel, plan)
    names = ["naive_persist", "naive_seasonal"] + (
        list(build_model_zoo()) if candidate_names is None else candidate_names
    )
    names = list(dict.fromkeys(names))
    rows = []
    for name in names:
        print(f"Validation: {name}", flush=True)
        started = time.perf_counter()
        row = {"model": name, "kind": "level" if name == "naive_seasonal" else "delta"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            errors = recursive_errors(
                panel[panel["period"] <= plan.validation_end], name,
                plan.validation_origins, plan.horizons, min(plan.validation_origins) + 1, plan.validation_end,
            )
            row.update(error_metrics(errors, "val"))
            row["warnings"] = "; ".join(sorted({str(w.message) for w in caught}))
        if set(errors["horizon"]) != set(range(1, plan.horizons + 1)):
            raise ValueError(f"{name}: incomplete validation horizons")
        row["fit_and_forecast_seconds"] = time.perf_counter() - started
        rows.append(row)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            errors.to_csv(out_dir / f"validation_{name.replace('@', '_')}.csv", index=False)
    scores = pd.DataFrame(rows)
    winner = choose_winner(scores)
    winner["evaluation_plan"] = asdict(plan)
    if out_dir:
        scores.to_csv(out_dir / "bakeoff_metrics.csv", index=False)
        (out_dir / "winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    return winner, scores
