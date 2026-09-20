"""Rebuild every figure used by the paper from the published result tables.

Run: ``python docs/build/figures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
WINNER = "huber_delta@location"

INK = "#0f2740"
ACCENT = "#c2410c"
MUTED = "#94a3b8"
BASELINE = "#0f766e"
FAMILY_COLOURS = {
    "baseline": BASELINE,
    "linear": "#1d4ed8",
    "robust": ACCENT,
    "trees": "#7c3aed",
}
PRETTY = {
    "naive_persist": "Last-quarter persistence",
    "naive_seasonal": "Seasonal persistence",
    "lin": "Least squares",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic Net",
    "huber": "Huber",
    "bayes": "Bayesian Ridge",
    "hgb": "Hist gradient boosting",
    "rf": "Random forest",
    "et": "Extra Trees",
    "gbr": "Gradient boosting",
    "xgb_d3": "XGBoost (depth 3)",
    "xgb_d6": "XGBoost (depth 6)",
}


def _family(name: str) -> str:
    if name.startswith("naive"):
        return "baseline"
    if name.startswith("huber"):
        return "robust"
    if name.split("_")[0] in {"lin", "ridge", "lasso", "enet", "bayes"}:
        return "linear"
    return "trees"


def _label(name: str) -> str:
    if name in PRETTY:
        return PRETTY[name]
    stem, _, features = name.partition("_delta@")
    return f"{PRETTY.get(stem, stem)} · {features}"


def _style(ax, grid_axis: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis=grid_axis, linestyle=":", alpha=0.45, color=MUTED)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(path)
    return path


def validation_ranking() -> None:
    scores = pd.read_csv(RESULTS / "bakeoff_metrics.csv").sort_values("val_mae", ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    colours = [FAMILY_COLOURS[_family(name)] for name in scores["model"]]
    bars = ax.barh([_label(name) for name in scores["model"]], scores["val_mae"], color=colours, height=0.72)
    for bar, value in zip(bars, scores["val_mae"]):
        ax.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8, color=INK)
    persistence = float(scores.loc[scores["model"] == "naive_persist", "val_mae"].iloc[0])
    ax.axvline(persistence, color=BASELINE, linestyle="--", linewidth=1.2)
    ax.annotate("persistence baseline", xy=(persistence, len(scores) - 0.2), xytext=(persistence + 0.4,
                len(scores) - 0.2), color=BASELINE, fontsize=8.5, va="center")
    ax.set_xlim(0, float(scores["val_mae"].max()) * 1.12)
    ax.set_xlabel("Equal-horizon mean validation MAE (AUD / week) — lower is better")
    ax.set_title("26 candidates, one recursive eight-quarter protocol", color=INK, fontsize=13, loc="left",
                 pad=26)
    handles = [plt.Line2D([], [], color=colour, linewidth=7) for colour in FAMILY_COLOURS.values()]
    ax.legend(handles, ["baseline", "linear", "robust linear", "trees / boosting"], frameon=False, fontsize=9,
              loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4)
    _style(ax)
    _save(fig, "validation_mae.png")


def horizon_errors() -> None:
    val = pd.read_csv(RESULTS / "validation_horizons.csv")
    bands = pd.read_csv(RESULTS / "residual_bands.csv")
    test = pd.read_csv(RESULTS / "test_horizons.csv")
    winner = val[val["model"] == WINNER]
    persist = val[val["model"] == "naive_persist"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    ax.fill_between(bands["horizon"], 0, bands["p80"], color=MUTED, alpha=0.18,
                    label="calibrated 80th-pct half-width")
    ax.plot(bands["horizon"], bands["p80"], color="#475569", linestyle="--", marker="s", markersize=4,
            linewidth=1.3)
    ax.plot(winner["horizon"], winner["val_mae"], color=INK, marker="o", markersize=4, label="Huber (validation)")
    ax.plot(persist["horizon"], persist["val_mae"], color=BASELINE, marker="o", markersize=4,
            label="persistence (validation)")
    ax.plot(test["horizon"], test["mae"], color=ACCENT, marker="D", markersize=5, linewidth=1.8,
            label="Huber (2025 audit)")
    ax.set_xlabel("Forecast horizon (quarters ahead)")
    ax.set_ylabel("AUD / week")
    ax.set_title("Error compounds with horizon", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax, grid_axis="both")

    ax = axes[1]
    ax.bar(test["horizon"], test["coverage"] * 100, color=ACCENT, alpha=0.85, width=0.55)
    ax.axhline(80, color=INK, linestyle="--", linewidth=1.2)
    ax.text(0.6, 81.5, "80% nominal", fontsize=8.5, color=INK)
    for horizon, coverage in zip(test["horizon"], test["coverage"]):
        ax.text(horizon, coverage * 100 + 1.2, f"{coverage * 100:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_xticks(list(test["horizon"]))
    ax.set_xlabel("Horizon (observed test quarters only)")
    ax.set_ylabel("Interval coverage (%)")
    ax.set_title("Bands were conservative in 2025", color=INK, fontsize=12, loc="left")
    _style(ax, grid_axis="y")
    _save(fig, "horizon_errors.png")


def origin_variation() -> None:
    origins = pd.read_csv(RESULTS / "validation_origins.csv")
    keep = {WINNER: ("Huber / location", ACCENT), "naive_persist": ("Persistence", BASELINE),
            "xgb_d6_delta@location": ("XGBoost depth 6", "#7c3aed")}
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    labels = [f"{(o - 1) // 4} Q{(o - 1) % 4 + 1}" for o in sorted(origins["origin"].unique())]
    width = 0.26
    for offset, (model, (label, colour)) in enumerate(keep.items()):
        subset = origins[origins["model"] == model].sort_values("origin")
        positions = [i + (offset - 1) * width for i in range(len(subset))]
        ax.bar(positions, subset["val_mae"], width=width, label=label, color=colour)
        for x, value in zip(positions, subset["val_mae"]):
            ax.text(x, value + 0.4, f"{value:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Forecast origin")
    ax.set_ylabel("Equal-horizon MAE (AUD / week)")
    ax.set_title("The winner does not win at every origin", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, grid_axis="y")
    _save(fig, "origin_variation.png")


def slice_errors() -> None:
    beds = pd.read_csv(RESULTS / "test_by_bedrooms.csv")
    types = pd.read_csv(RESULTS / "test_by_apartment_type.csv")
    groups = pd.read_csv(RESULTS / "test_by_suburb_group.csv").sort_values("mae")
    extremes = pd.concat([groups.head(5), groups.tail(5)])
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [1, 0.75, 1.5]})

    ax = axes[0]
    ax.bar(beds["bedrooms"].astype(str), beds["mae"], color=ACCENT, alpha=0.85)
    for x, (value, n) in enumerate(zip(beds["mae"], beds["n"])):
        ax.text(x, value + 0.5, f"{value:.1f}\nn={n}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, beds["mae"].max() * 1.3)
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Audit MAE (AUD / week)")
    ax.set_title("Large dwellings are harder", color=INK, fontsize=11.5, loc="left")
    _style(ax, grid_axis="y")

    ax = axes[1]
    ax.bar(types["apartment_type"], types["mae"], color=INK, alpha=0.85)
    for x, (value, n) in enumerate(zip(types["mae"], types["n"])):
        ax.text(x, value + 0.3, f"{value:.1f}\nn={n}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, types["mae"].max() * 1.3)
    ax.set_xlabel("Dwelling type")
    ax.set_title("Houses over flats", color=INK, fontsize=11.5, loc="left")
    _style(ax, grid_axis="y")

    ax = axes[2]
    colours = [BASELINE] * 5 + [ACCENT] * 5
    ax.barh(extremes["suburb_group"], extremes["mae"], color=colours)
    ax.invert_yaxis()
    ax.set_xlabel("Audit MAE (AUD / week), 18 pairs each")
    ax.set_title("Best and worst five suburb groups", color=INK, fontsize=11.5, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    _style(ax)
    _save(fig, "slice_errors.png")


def protocol_illusions() -> None:
    data = json.loads((RESULTS / "evaluation_illusions.json").read_text(encoding="utf-8"))
    labels = [
        "Shuffled 80/20 split\none step, observed lags",
        "Temporal split\none step, observed lags",
        "Recursive 8 quarters\nself-fed lags (shipped)",
    ]
    values = [
        data["shuffled_split_one_step"]["mae"],
        data["temporal_split_one_step"]["mae"],
        data["recursive_eight_horizon"]["mae"],
    ]
    notes = [
        f"R² = {data['shuffled_split_one_step']['r2_on_level']:.3f}",
        f"R² = {data['temporal_split_one_step']['r2_on_level']:.3f}",
        f"h=1: {data['recursive_eight_horizon']['mae_horizon_1']:.1f}  "
        f"h=8: {data['recursive_eight_horizon']['mae_horizon_8']:.1f}",
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    bars = ax.bar(labels, values, color=[MUTED, "#64748b", ACCENT], width=0.6)
    for bar, value, note in zip(bars, values, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"MAE {value:.2f}", ha="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, value / 2, note, ha="center", fontsize=8.5, color="white")
    ax.set_ylabel("MAE (AUD / week)")
    ax.set_title("Same model, same features — three evaluation protocols", color=INK, fontsize=12, loc="left")
    _style(ax, grid_axis="y")
    _save(fig, "protocol_illusions.png")


def evaluation_timeline() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 3.4))
    spans = [
        ("Training history\n2001 Q1 – 2018 Q4", 2001, 2018.75, "#dbeafe", 0, "in"),
        ("Validation targets\n2019 Q1 – 2023 Q4", 2019.0, 2024.0, "#bfdbfe", 0, "in"),
        ("Calibration\n2024", 2024.0, 2025.0, "#fed7aa", (2022.6, -0.62), "below"),
        ("Frozen audit\n2025 Q1–Q3", 2025.0, 2025.75, "#fecaca", (2025.0, -1.08), "below"),
        ("Deployment forecast\n2025 Q4 – 2027 Q3", 2025.75, 2027.75, "#e2e8f0", (2027.4, -0.62), "below"),
    ]
    for label, start, end, colour, offset, placement in spans:
        ax.barh(0, end - start, left=start, height=0.5, color=colour, edgecolor="white")
        middle = (start + end) / 2
        if placement == "in":
            ax.text(middle, 0, label, ha="center", va="center", fontsize=8.5, color=INK)
        else:
            ax.annotate(label, xy=(middle, -0.26), xytext=offset, ha="center", va="top", fontsize=8.5,
                        color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9})
    for origin in (2018.75, 2019.75, 2020.75, 2021.75):
        ax.plot([origin], [0.42], marker="v", color=INK, markersize=6)
    ax.text(2020.25, 0.62, "validation origins (expanding window)", ha="center", fontsize=8, color=INK)
    ax.plot([2024.75], [0.42], marker="v", color=ACCENT, markersize=7)
    ax.text(2024.75, 0.86, "audit origin", ha="center", fontsize=8, color=ACCENT)
    ax.set_xlim(2000.5, 2029.6)
    ax.set_ylim(-1.5, 1.0)
    ax.set_yticks([])
    ax.set_xticks([2001, 2005, 2010, 2015, 2019, 2022, 2024, 2026, 2027])
    ax.set_xticklabels(["2001", "2005", "2010", "2015", "2019", "2022", "2024", "2026", "2027"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("One chronological pass: select, calibrate, audit, deploy", color=INK, fontsize=12, loc="left")
    _save(fig, "evaluation_timeline.png")


def forecast_example() -> None:
    panel = pd.read_csv(ROOT / "data" / "rent_panel.csv")
    forecasts = pd.read_csv(RESULTS / "deployment_forecasts.csv")
    series = ("Brunswick", "flat", 2)
    history = panel[
        (panel["suburb_group"] == series[0])
        & (panel["apartment_type"] == series[1])
        & (panel["bedrooms"] == series[2])
        & (panel["period"] >= 2015 * 4)
    ].sort_values("period")
    future = forecasts[
        (forecasts["suburb_group"] == series[0])
        & (forecasts["apartment_type"] == series[1])
        & (forecasts["bedrooms"] == series[2])
    ].sort_values("horizon")
    hx = (history["period"] - 1) // 4 + ((history["period"] - 1) % 4) / 4
    anchor = history["period"].max()
    fx = [(anchor + h - 1) // 4 + ((anchor + h - 1) % 4) / 4 for h in future["horizon"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(hx, history["median"], color=INK, linewidth=1.8, label="observed moving-annual median")
    ax.fill_between(fx, future["lower"], future["upper"], color=ACCENT, alpha=0.15,
                    label="80th-percentile empirical band")
    ax.plot(fx, future["prediction"], color=ACCENT, marker="o", markersize=4, linewidth=1.8, label="forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("AUD / week")
    ax.set_title("Brunswick, 2-bedroom flat: shipped forecast to 2027 Q3", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, grid_axis="both")
    _save(fig, "forecast_example.png")


def social_card() -> None:
    """1200x627 share image for LinkedIn / OG previews."""
    illusions = json.loads((RESULTS / "evaluation_illusions.json").read_text())
    fig = plt.figure(figsize=(12, 6.27), dpi=100)
    fig.patch.set_facecolor(INK)
    fig.text(0.055, 0.86, "V I C T O R I A N   R E N T A L   F O R E C A S T I N G", color="#f6b48a",
             fontsize=11, fontweight="bold")
    fig.text(0.055, 0.70, "Same model.\nThree ways to score it.", color="white", fontsize=38,
             fontweight="bold", linespacing=1.15, va="top")
    fig.text(0.055, 0.32, "26 candidates · 862 rental series\none recursive eight-quarter protocol",
             color="#bcd2e6", fontsize=14, linespacing=1.5, va="top")
    fig.text(0.055, 0.17, "github.com/kelvinvalani/vic-rent-ml", color="#f6b48a", fontsize=13)

    ax = fig.add_axes((0.56, 0.16, 0.39, 0.66))
    ax.set_facecolor(INK)
    values = [
        illusions["shuffled_split_one_step"]["mae"],
        illusions["temporal_split_one_step"]["mae"],
        illusions["recursive_eight_horizon"]["mae"],
    ]
    labels = ["shuffled\nsplit", "temporal,\none step", "recursive\n(shipped)"]
    colours = ["#64748b", "#94a3b8", ACCENT]
    bars = ax.bar(labels, values, color=colours, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.9, f"${value:.2f}", ha="center",
                color="white", fontsize=17, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.26)
    ax.set_yticks([])
    ax.tick_params(colors="#bcd2e6", labelsize=12, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#33587f")
    ax.set_title("Mean absolute error, AUD / week", color="#bcd2e6", fontsize=12, loc="left", pad=12)
    path = FIGURES / "social_card.png"
    fig.savefig(path, facecolor=INK)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    validation_ranking()
    horizon_errors()
    origin_variation()
    slice_errors()
    protocol_illusions()
    evaluation_timeline()
    forecast_example()
    social_card()


if __name__ == "__main__":
    main()
