"""Select, calibrate, audit once, and refit the same candidate for deployment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .bakeoff import (
    EvaluationPlan, choose_winner, error_metrics, fit_candidate, recursive_errors, run_bakeoff, validate_plan,
)
from .recipe import KEY_COLS, band_table_from_errors, forecast_panel


def panel_digest(panel: pd.DataFrame) -> str:
    canonical = panel.sort_values(KEY_COLS + ["period"]).sort_index(axis=1).to_csv(
        index=False, lineterminator="\n"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def source_digest() -> str:
    return hashlib.sha256(b"".join(
        (Path(__file__).parent / filename).read_bytes().replace(b"\r\n", b"\n")
        for filename in ("bakeoff.py", "recipe.py", "ingest.py", "training.py", "config.py")
    )).hexdigest()


def evaluate(panel: pd.DataFrame, out_dir: Path, plan: EvaluationPlan | None = None, candidate_names=None) -> dict:
    plan = plan or EvaluationPlan()
    winner, metrics = run_bakeoff(panel, out_dir=out_dir, plan=plan, candidate_names=candidate_names)
    winner["selection_data_sha256"] = panel_digest(panel[panel["period"] <= plan.validation_end])
    winner["source_sha256"] = source_digest()
    winner["candidate_names"] = metrics["model"].tolist()
    (out_dir / "winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    return winner


def train(
    panel: pd.DataFrame, out_dir: Path, *, reuse_selection: bool = False,
    plan: EvaluationPlan | None = None, candidate_names=None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if reuse_selection:
        winner = json.loads((out_dir / "winner.json").read_text(encoding="utf-8"))
        plan_values = winner["evaluation_plan"].copy()
        plan_values["validation_origins"] = tuple(plan_values["validation_origins"])
        saved_plan = EvaluationPlan(**plan_values)
        if plan is not None and plan != saved_plan:
            raise ValueError("Evaluation plan changed; rerun selection")
        plan = saved_plan
        validate_plan(panel, plan)
        metrics = pd.read_csv(out_dir / "bakeoff_metrics.csv")
        if winner["model"] != choose_winner(metrics)["model"]:
            raise ValueError("Winner does not match saved validation scores")
        if winner["source_sha256"] != source_digest():
            raise ValueError("Training source changed; rerun selection")
        if winner["selection_data_sha256"] != panel_digest(panel[panel["period"] <= plan.validation_end]):
            raise ValueError("Selection data changed; rerun selection")
    else:
        plan = plan or EvaluationPlan()
        winner = evaluate(panel, out_dir, plan, candidate_names)
    name = winner["model"]
    print(f"Frozen winner: {name}. Calibrating on held-out targets.", flush=True)
    calibration = recursive_errors(
        panel[panel["period"] <= plan.test_origin], name,
        list(range(plan.calibration_start - plan.horizons, plan.test_origin)),
        plan.horizons, plan.calibration_start, plan.test_origin,
    )
    table, bands = band_table_from_errors(calibration)
    if set(table["horizon"]) != set(range(1, plan.horizons + 1)):
        raise ValueError("Calibration does not cover every supported horizon")
    bands["recipe"] = name
    calibration.to_csv(out_dir / "calibration_errors.csv", index=False)
    table.to_csv(out_dir / "residual_bands.csv", index=False)
    (out_dir / "residual_bands.json").write_text(json.dumps(bands, indent=2), encoding="utf-8")

    audit_rows, audit_errors = [], []
    for candidate in dict.fromkeys([name, "naive_persist", "naive_seasonal"]):
        print(f"Untouched test: {candidate}", flush=True)
        errors = recursive_errors(
            panel, candidate, [plan.test_origin], plan.horizons, plan.test_origin + 1, int(panel["period"].max())
        )
        errors["model"] = candidate
        audit_errors.append(errors)
        audit_rows.append({"model": candidate, **error_metrics(errors, "test")})
        if candidate == name:
            errors["half_width"] = errors["horizon"].map(bands["half_width_p80"])
            errors["covered"] = errors["abs_err"] <= errors["half_width"]
            diagnostics = errors.groupby("horizon").agg(
                n=("abs_err", "size"), mae=("abs_err", "mean"), coverage=("covered", "mean"),
                width=("half_width", "mean"),
            ).reset_index()
            diagnostics.to_csv(out_dir / "test_horizons.csv", index=False)
            for slice_col in ("suburb_group", "apartment_type", "bedrooms"):
                errors.groupby(slice_col).agg(n=("abs_err", "size"), mae=("abs_err", "mean")).to_csv(
                    out_dir / f"test_by_{slice_col}.csv"
                )
    pd.DataFrame(audit_rows).to_csv(out_dir / "test_metrics.csv", index=False)
    pd.concat(audit_errors, ignore_index=True).to_csv(out_dir / "test_errors.csv", index=False)
    print(f"Refitting {name} on all {len(panel):,} rows.", flush=True)
    ship = fit_candidate(panel, name)
    origin = int(panel["period"].max())
    history = panel[panel["period"] >= origin - 3].copy()
    forecasts = forecast_panel(ship, history, origin, plan.horizons, kind=winner["kind"])
    metadata = {
        "schema_version": 2, "kind": winner["kind"], "recipe": name, "uses_count": False,
        "trained_through": {"year": (origin - 1) // 4, "quarter": (origin - 1) % 4 + 1},
        "winner": winner, "residual_bands": bands, "test_metrics": audit_rows,
        "data_sha256": panel_digest(panel), "source_sha256": source_digest(),
        "python": platform.python_version(),
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
        },
        "rows": len(panel), "series": len(panel[KEY_COLS].drop_duplicates()),
        "evaluation_plan": asdict(plan), "max_horizon": plan.horizons,
    }
    payload = {**metadata, "model": ship, "history": history, "forecasts": forecasts}
    artifact = out_dir / "production_clone.joblib"
    joblib.dump(payload, artifact, compress=3)
    metadata["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not np.isfinite(forecasts["prediction"]).all():
        raise ValueError("Non-finite deployment forecasts")
    return payload
