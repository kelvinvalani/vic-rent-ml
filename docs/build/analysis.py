"""Recompute the published audit and add uncertainty analysis for the paper.

Outputs (all under docs/results/):

* ``audit_uncertainty.json``  – clustered bootstrap intervals for the winner's
  advantage over last-quarter persistence, plus per-series win rates.
* ``deployment_forecasts.csv`` – the shipped recursive forecast for every series,
  2025 Q4 through 2027 Q3, with the calibrated 80th-percentile band.
* ``reproduction.json`` – dependency versions used here and the audit metrics
  they produce, so a reader can check cross-version agreement with the
  published run.

Run: ``python docs/build/analysis.py``
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vic_rent_ml.bakeoff import EvaluationPlan, error_metrics, fit_candidate, recursive_errors
from vic_rent_ml.recipe import KEY_COLS, forecast_panel

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
PANEL = ROOT / "data" / "rent_panel.csv"
WINNER = "huber_delta@location"
BASELINE = "naive_persist"
BOOTSTRAP_DRAWS = 2000
SEED = 42


def _period_label(period: int) -> str:
    return f"{(period - 1) // 4} Q{(period - 1) % 4 + 1}"


def _equal_horizon_mae(frame: pd.DataFrame) -> float:
    return float(frame.groupby("horizon")["abs_err"].mean().mean())


def cluster_bootstrap(paired: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> dict:
    """Bootstrap the equal-horizon MAE gap, resampling whole suburb groups.

    Forecast errors are correlated within a suburb group and across horizons of
    the same trajectory, so rows are not independent. Resampling clusters keeps
    that dependence inside the resampled unit.
    """
    rng = np.random.default_rng(seed)
    clusters = paired["suburb_group"].to_numpy()
    unique = np.unique(clusters)
    index = {name: np.flatnonzero(clusters == name) for name in unique}
    gaps = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index[name] for name in picked])
        sample = paired.iloc[rows]
        by_horizon = sample.groupby("horizon")[["abs_err_winner", "abs_err_baseline"]].mean()
        gaps[draw] = float((by_horizon["abs_err_baseline"] - by_horizon["abs_err_winner"]).mean())
    observed = float(
        paired.groupby("horizon")["abs_err_baseline"].mean().mean()
        - paired.groupby("horizon")["abs_err_winner"].mean().mean()
    )
    return {
        "observed_mae_gap": observed,
        "ci95_low": float(np.quantile(gaps, 0.025)),
        "ci95_high": float(np.quantile(gaps, 0.975)),
        "share_of_draws_favouring_winner": float((gaps > 0).mean()),
        "clusters": int(unique.size),
        "draws": draws,
    }


def paired_errors(panel: pd.DataFrame, origins, horizons: int, target_min: int, target_max: int) -> pd.DataFrame:
    keys = KEY_COLS + ["period", "horizon", "origin"]
    winner = recursive_errors(panel, WINNER, origins, horizons, target_min, target_max)
    baseline = recursive_errors(panel, BASELINE, origins, horizons, target_min, target_max)
    merged = winner[keys + ["abs_err"]].merge(
        baseline[keys + ["abs_err"]], on=keys, suffixes=("_winner", "_baseline"), validate="one_to_one"
    )
    return merged


def window_report(panel: pd.DataFrame, origins, horizons, target_min, target_max, label: str) -> dict:
    paired = paired_errors(panel, origins, horizons, target_min, target_max)
    series_gap = paired.groupby(KEY_COLS)[["abs_err_winner", "abs_err_baseline"]].mean()
    beaten = series_gap["abs_err_winner"] < series_gap["abs_err_baseline"]
    report = {
        "window": label,
        "origins": [_period_label(int(o)) for o in origins],
        "targets": f"{_period_label(target_min)} – {_period_label(target_max)}",
        "pairs": int(len(paired)),
        "series": int(len(series_gap)),
        "winner_mae": _equal_horizon_mae(paired.rename(columns={"abs_err_winner": "abs_err"})),
        "baseline_mae": _equal_horizon_mae(paired.rename(columns={"abs_err_baseline": "abs_err"})),
        "series_win_rate": float(beaten.mean()),
        "bootstrap": cluster_bootstrap(paired),
    }
    report["relative_improvement_pct"] = 100.0 * (
        (report["baseline_mae"] - report["winner_mae"]) / report["baseline_mae"]
    )
    return report


def deployment_forecasts(panel: pd.DataFrame, plan: EvaluationPlan) -> pd.DataFrame:
    bands = json.loads((RESULTS / "residual_bands.json").read_text(encoding="utf-8"))["half_width_p80"]
    model = fit_candidate(panel, WINNER)
    origin = int(panel["period"].max())
    history = panel[panel["period"] >= origin - 3].copy()
    forecast = forecast_panel(model, history, origin, plan.horizons, kind="delta")
    anchor = panel[panel["period"] == origin][KEY_COLS + ["median", "postcodes"]]
    forecast = forecast.merge(anchor.rename(columns={"median": "anchor_median"}), on=KEY_COLS, how="left")
    forecast["half_width"] = forecast["horizon"].map(lambda h: bands[str(h)])
    forecast["lower"] = forecast["prediction"] - forecast["half_width"]
    forecast["upper"] = forecast["prediction"] + forecast["half_width"]
    forecast["label"] = forecast["period"].map(_period_label)
    columns = KEY_COLS + ["postcodes", "horizon", "label", "anchor_median", "prediction", "lower", "upper"]
    return forecast[columns].round(2)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL)
    plan = EvaluationPlan()

    validation = window_report(
        panel[panel["period"] <= plan.validation_end],
        list(plan.validation_origins),
        plan.horizons,
        min(plan.validation_origins) + 1,
        plan.validation_end,
        "validation (2018 Q4 – 2021 Q4 origins)",
    )
    audit = window_report(
        panel,
        [plan.test_origin],
        plan.horizons,
        plan.test_origin + 1,
        int(panel["period"].max()),
        "frozen-winner audit (2024 Q4 origin)",
    )

    payload = {
        "winner": WINNER,
        "baseline": BASELINE,
        "metric": "equal-horizon mean MAE (AUD/week)",
        "note": (
            "Bootstrap resamples whole suburb groups with replacement; intervals describe "
            "sampling variability of the measured gap on this panel, not out-of-sample guarantees."
        ),
        "windows": [validation, audit],
    }
    (RESULTS / "audit_uncertainty.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    audit_errors = recursive_errors(
        panel, WINNER, [plan.test_origin], plan.horizons, plan.test_origin + 1, int(panel["period"].max())
    )
    reproduction = {
        "python": platform.python_version(),
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
        },
        "audit_metrics": error_metrics(audit_errors, "test"),
        "published_audit_metrics": json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))[
            "test_metrics"
        ][0],
    }
    (RESULTS / "reproduction.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")

    forecasts = deployment_forecasts(panel, plan)
    forecasts.to_csv(RESULTS / "deployment_forecasts.csv", index=False)
    print(json.dumps(payload, indent=2))
    print(reproduction)
    print(forecasts.head())


if __name__ == "__main__":
    main()
