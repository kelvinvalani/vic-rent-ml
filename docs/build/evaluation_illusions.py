"""Quantify how optimistic the wrong evaluation protocols look on this panel.

Three scores for the *same* estimator and features (``huber_delta@location``):

1. shuffled 80/20 split, scored one step ahead with observed lags;
2. temporal split, scored one step ahead with observed lags (teacher forcing);
3. the protocol this project ships: recursive eight-quarter forecasts.

Writes ``docs/results/evaluation_illusions.json``.

Run: ``python docs/build/evaluation_illusions.py``
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from vic_rent_ml.bakeoff import EvaluationPlan, build_model_zoo, recursive_errors
from vic_rent_ml.recipe import production_feature_frame

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
CANDIDATE = "huber_delta@location"
SEED = 42


def _fit(train: pd.DataFrame):
    _, template = build_model_zoo()[CANDIDATE]
    from sklearn.base import clone

    return clone(template).fit(production_feature_frame(train), train["median"] - train["median_lag_1"])


def _one_step_scores(model, frame: pd.DataFrame) -> dict:
    predicted = model.predict(production_feature_frame(frame)) + frame["median_lag_1"].to_numpy()
    actual = frame["median"].to_numpy()
    residual = predicted - actual
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2_on_level": 1.0 - ss_res / ss_tot,
        "n": int(len(frame)),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    panel = pd.read_csv(ROOT / "data" / "rent_panel.csv")
    plan = EvaluationPlan()
    selection_panel = panel[panel["period"] <= plan.validation_end]

    train, test = train_test_split(selection_panel, test_size=0.2, random_state=SEED, shuffle=True)
    shuffled = _one_step_scores(_fit(train), test)

    cutoff = max(plan.validation_origins)
    temporal_train = selection_panel[selection_panel["period"] <= cutoff]
    temporal_test = selection_panel[selection_panel["period"] > cutoff]
    teacher_forced = _one_step_scores(_fit(temporal_train), temporal_test)

    recursive = recursive_errors(
        selection_panel,
        CANDIDATE,
        list(plan.validation_origins),
        plan.horizons,
        min(plan.validation_origins) + 1,
        plan.validation_end,
    )
    by_horizon = recursive.groupby("horizon")["abs_err"].mean()
    protocol = {
        "shuffled_split_one_step": shuffled,
        "temporal_split_one_step": teacher_forced,
        "recursive_eight_horizon": {
            "mae": float(by_horizon.mean()),
            "mae_horizon_1": float(by_horizon.loc[1]),
            "mae_horizon_8": float(by_horizon.loc[8]),
            "n": int(len(recursive)),
        },
        "note": (
            "Same estimator, same features. Only the evaluation protocol changes. "
            "Shuffled splits let overlapping moving-annual windows and neighbouring "
            "quarters of the same series appear on both sides of the split; one-step "
            "scoring hands the model an observed lag it will not have in production."
        ),
    }
    (RESULTS / "evaluation_illusions.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
