"""Draw validation and residual-band figures from the published result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def plot_validation_ranking() -> Path:
    scores = pd.read_csv(RESULTS / "bakeoff_metrics.csv").sort_values("val_mae")
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    colors = ["#1f4e79" if "huber_delta@location" in name else "#8aa2c2" for name in scores["model"]]
    ax.barh(scores["model"], scores["val_mae"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Equal-horizon mean validation MAE (AUD / week)")
    ax.set_title("Recursive eight-horizon bake-off")
    _style(ax)
    fig.tight_layout()
    path = FIGURES / "validation_mae.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_horizon_errors() -> Path:
    val = pd.read_csv(RESULTS / "validation_horizons.csv")
    bands = pd.read_csv(RESULTS / "residual_bands.csv")
    test = pd.read_csv(RESULTS / "test_horizons.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    winner = val[val["model"] == "huber_delta@location"]
    persist = val[val["model"] == "naive_persist"]
    ax.plot(winner["horizon"], winner["val_mae"], marker="o", label="Huber / location (validation)")
    ax.plot(persist["horizon"], persist["val_mae"], marker="o", label="Last-quarter persistence (validation)")
    ax.plot(bands["horizon"], bands["p80"], marker="s", linestyle="--", label="Calibrated 80th-pct band width")
    ax.plot(test["horizon"], test["mae"], marker="D", label="Huber audit MAE (2025 Q1–Q3)")
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("Weekly dollars")
    ax.set_title("Error grows with horizon; bands are measured, not ±8%")
    ax.legend(frameon=False)
    _style(ax)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    path = FIGURES / "horizon_errors.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(plot_validation_ranking())
    print(plot_horizon_errors())


if __name__ == "__main__":
    main()
