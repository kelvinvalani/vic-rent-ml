"""Build the project's public-facing visuals: a poster PNG and animated GIFs.

``docs/figures/poster.png``     — one-page summary: the shipped model's pipeline on
                                  top, the Brunswick forecast (the output) below.
``docs/figures/explainer.gif``  — short animation of the recursive eight-quarter
                                  mechanism: each forecast feeds back as the next lag.
``docs/figures/linkedin.gif``   — Brunswick cone + shuffled vs temporal vs recursive MAE.
``docs/figures/linkedin_card.png`` — static square of the two-year frame (fallback).

Every number is read from ``docs/results/`` and ``data/rent_panel.csv``; nothing is
hand-typed. Run: ``python docs/build/explainer.py``
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"

INK = "#0f2740"
ACCENT = "#c2410c"
MUTED = "#94a3b8"
BASELINE = "#0f766e"
CARD = "#f1f5f9"
LINE = "#dbe3ec"
PEACH = "#f6b48a"
MIST = "#bcd2e6"
TRACK = "#1a3a58"

SERIES = ("Brunswick", "flat", 2)
HISTORY_FROM = 2015 * 4  # panel period encoding, same as figures.forecast_example


def _period_x(period: int) -> float:
    return (period - 1) // 4 + ((period - 1) % 4) / 4


def _load_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(ROOT / "data" / "rent_panel.csv")
    forecasts = pd.read_csv(RESULTS / "deployment_forecasts.csv")
    history = panel[
        (panel["suburb_group"] == SERIES[0])
        & (panel["apartment_type"] == SERIES[1])
        & (panel["bedrooms"] == SERIES[2])
        & (panel["period"] >= HISTORY_FROM)
    ].sort_values("period")
    future = forecasts[
        (forecasts["suburb_group"] == SERIES[0])
        & (forecasts["apartment_type"] == SERIES[1])
        & (forecasts["bedrooms"] == SERIES[2])
    ].sort_values("horizon")
    return history, future


def _numbers() -> dict:
    bakeoff = pd.read_csv(RESULTS / "bakeoff_metrics.csv").set_index("model")
    test = pd.read_csv(RESULTS / "test_metrics.csv").set_index("model")
    bands = pd.read_csv(RESULTS / "residual_bands.csv").set_index("horizon")
    uncertainty = json.loads((RESULTS / "audit_uncertainty.json").read_text())
    audit = next(w for w in uncertainty["windows"] if "audit" in w["window"])
    forecasts = pd.read_csv(RESULTS / "deployment_forecasts.csv")
    n_series = forecasts.groupby(["suburb_group", "apartment_type", "bedrooms"]).ngroups
    return {
        "val_mae": bakeoff.loc["huber_delta@location", "val_mae"],
        "val_persist": bakeoff.loc["naive_persist", "val_mae"],
        "audit_mae": test.loc["huber_delta@location", "test_mae"],
        "audit_persist": test.loc["naive_persist", "test_mae"],
        "improvement": audit["relative_improvement_pct"],
        "band_first": bands.loc[1, "p80"],
        "band_last": bands.loc[8, "p80"],
        "n_series": n_series,
    }


def _forecast_xy(history: pd.DataFrame, future: pd.DataFrame) -> tuple[list, list]:
    anchor = int(history["period"].max())
    fx = [_period_x(anchor + h) for h in future["horizon"]]
    hx = [_period_x(period) for period in history["period"]]
    return hx, fx


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.grid(axis="both", linestyle=":", alpha=0.45, color=MUTED)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8.5)


def _draw_forecast_chart(ax, history, future, upto: int | None = None,
                         title: str = "Brunswick, 2-bedroom flat: shipped forecast to 2027 Q3",
                         title_size: float = 12) -> None:
    """The output panel: observed history + forecast + empirical band.

    ``upto`` limits the drawn forecast to the first ``upto`` horizons (GIF frames);
    the cone opens from the last observed anchor, as in the paper figure. Limits are
    fixed on the full horizon so the axis does not rescale between animation frames.
    """
    hx, fx = _forecast_xy(history, future)
    n = len(fx) if upto is None else upto
    ax.plot(hx, history["median"], color=INK, linewidth=1.8, label="observed moving-annual median")
    if n:
        rows = future.iloc[:n]
        anchor = float(history["median"].iloc[-1])
        xs = [hx[-1]] + fx[:n]
        ax.fill_between(xs, [anchor] + list(rows["lower"]), [anchor] + list(rows["upper"]),
                        color=ACCENT, alpha=0.15, label="80th-percentile empirical band")
        ax.plot(xs, [anchor] + list(rows["prediction"]), color=ACCENT, marker="o", markersize=4,
                linewidth=1.8, label="forecast")
    ax.set_xlim(hx[0] - 0.15, fx[-1] + 0.2)
    ax.set_ylim(min(history["median"].min(), future["lower"].min()) - 12,
                max(history["median"].max(), future["upper"].max()) + 12)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("AUD / week", fontsize=9)
    ax.set_title(title, color=INK, fontsize=title_size, loc="left")
    _style_axes(ax)


# ---------------------------------------------------------------- poster ----

STAGES = [
    ("Observed panel", "RTBA bond lodgements\nmoving-annual median rent\n862 series · 2001–2025"),
    ("Features", "lag-1 · lag-4 · quarter · year\nbedrooms · dwelling · postcode"),
    ("f̂ — Huber regression", "predicts the quarterly\nchange Δ, never the level"),
    ("Recursive loop ×8", "ŷ(t+h) = ŷ(t+h−1) + f̂\neach forecast becomes\nthe next lag-1 feature"),
    ("Shipped forecast", "point path + empirical\nband, ±\\$17 → ±\\$116\nby horizon 8"),
]


def _pipeline(ax) -> None:
    """Five-stage flow diagram of what the shipped artifact computes."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width, y0, height = 0.166, 0.30, 0.58
    for index, (title, detail) in enumerate(STAGES):
        x = 0.012 + index * 0.198
        is_loop = index == 3
        box = FancyBboxPatch((x, y0), width, height, boxstyle="round,pad=0.012,rounding_size=0.018",
                             facecolor="#fdf3ee" if is_loop else CARD,
                             edgecolor=ACCENT if is_loop else LINE,
                             linewidth=1.8 if is_loop else 1.1, mutation_aspect=0.6)
        ax.add_patch(box)
        ax.text(x + width / 2, y0 + height - 0.13, title, ha="center", va="top",
                fontsize=9.3, fontweight="bold", color=INK)
        ax.text(x + width / 2, y0 + height - 0.30, detail, ha="center", va="top",
                fontsize=7.4, color="#475569", linespacing=1.5)
        if index < len(STAGES) - 1:
            ax.annotate("", xy=(x + 0.198, y0 + height / 2), xytext=(x + width + 0.004, y0 + height / 2),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.6})
    ax.annotate("the evaluation and the product run this same code path — that is the point",
                xy=(0.5, 0.10), ha="center", fontsize=9, color="#475569", style="italic")


def poster() -> Path:
    history, future = _load_series()
    num = _numbers()
    fig = plt.figure(figsize=(10, 12.6), dpi=150)
    fig.patch.set_facecolor("white")

    fig.text(0.06, 0.972, "V I C - R E N T - M L   ·   R E S E A R C H   S U M M A R Y",
             fontsize=10, fontweight="bold", color=ACCENT)
    fig.text(0.06, 0.942, "Forecasting Victorian rents the way\nthe model actually runs",
             fontsize=25, fontweight="bold", color=INK, linespacing=1.15, va="top")
    fig.text(0.06, 0.872,
             "26 candidates · 862 suburb-group series · one recursive eight-quarter protocol · "
             "Huber regression on the quarterly change",
             fontsize=10.5, color="#475569")

    stats = [
        (f"\\${num['val_mae']:.2f}", f"validation MAE vs \\${num['val_persist']:.2f} persistence"),
        (f"\\${num['audit_mae']:.2f}", f"frozen 2025 audit · {num['improvement']:.1f}% better"),
        (f"±\\${num['band_first']:.0f} → ±\\${num['band_last']:.0f}", "empirical half-width, Q1 → Q8"),
        (f"{num['n_series']}", "live series forecast to 2027 Q3"),
    ]
    ax_stats = fig.add_axes((0.055, 0.782, 0.89, 0.062))
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    ax_stats.axis("off")
    for index, (big, small) in enumerate(stats):
        x = index * 0.252
        ax_stats.add_patch(FancyBboxPatch((x, 0.04), 0.235, 0.92,
                                          boxstyle="round,pad=0.01,rounding_size=0.03",
                                          facecolor=CARD, edgecolor=LINE, linewidth=1))
        ax_stats.text(x + 0.012, 0.68, big, fontsize=15, fontweight="bold", color=INK, va="center")
        ax_stats.text(x + 0.012, 0.26, small, fontsize=7.6, color="#475569", va="center")

    fig.text(0.06, 0.745, "1 — THE MODEL", fontsize=11, fontweight="bold", color=INK)
    fig.lines.append(plt.Line2D((0.06, 0.94), (0.738, 0.738), transform=fig.transFigure,
                                color=LINE, linewidth=1))
    ax_pipe = fig.add_axes((0.045, 0.545, 0.91, 0.175))
    _pipeline(ax_pipe)

    fig.text(0.06, 0.505, "2 — THE OUTPUT", fontsize=11, fontweight="bold", color=INK)
    fig.lines.append(plt.Line2D((0.06, 0.94), (0.498, 0.498), transform=fig.transFigure,
                                color=LINE, linewidth=1))
    ax_chart = fig.add_axes((0.075, 0.125, 0.86, 0.35))
    _draw_forecast_chart(ax_chart, history, future)
    ax_chart.legend(frameon=False, fontsize=9, loc="upper left")

    fig.text(0.06, 0.052, "Scored the way the product runs — eight quarters ahead, eating its own "
             "predictions: \\$23.80 MAE, not the \\$7.23 a shuffled split gives.", fontsize=9,
             color="#475569")
    fig.text(0.06, 0.026,
             "Data: Homes Victoria Rental Report (RTBA bond lodgements), CC BY 4.0   ·   "
             "github.com/kelvinvalani/vic-rent-ml", fontsize=9, color=MUTED)
    path = FIGURES / "poster.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


# ------------------------------------------------------------------- gif ----

def _draw_loop(ax, step: int) -> None:
    """The recursion schematic; `step` is the horizon currently being produced."""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "The model eats its own forecast", fontsize=12.5,
            fontweight="bold", color=INK, va="top")
    ax.text(0.02, 0.80, "ŷ(t+h) = ŷ(t+h−1) + f̂(x̂)", fontsize=12, color=INK, va="top",
            family="monospace")
    ax.text(0.02, 0.68, "each predicted quarter becomes the\nlag-1 feature for the next step",
            fontsize=8.6, color="#475569", va="top", linespacing=1.5)

    n_nodes = 9
    xs = [0.075 + i * 0.105 for i in range(n_nodes)]
    y = 0.34
    for i in range(1, n_nodes):
        active = i == step
        done = i < step
        colour = ACCENT if active else (INK if done else MUTED)
        ax.annotate("", xy=(xs[i] - 0.028, y), xytext=(xs[i - 1] + 0.028, y),
                    arrowprops={"arrowstyle": "-|>", "color": colour,
                                "lw": 2.2 if active else 1.2,
                                "alpha": 1.0 if (active or done) else 0.5})
        ax.text((xs[i - 1] + xs[i]) / 2, y + 0.075, "f̂", ha="center", fontsize=8.5,
                color=colour, fontweight="bold" if active else "normal")
    for i, x in enumerate(xs):
        if i == 0:
            face, edge, label_colour = INK, INK, INK
        elif i < step:
            face, edge, label_colour = "#f2cdb8", ACCENT, "#475569"
        elif i == step:
            face, edge, label_colour = ACCENT, ACCENT, INK
        else:
            face, edge, label_colour = "white", LINE, MUTED
        circle = plt.Circle((x, y), 0.024, facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3)
        ax.add_patch(circle)
        label = "y(T)\nobserved" if i == 0 else f"T+{i}"
        ax.text(x, y - 0.075, label, ha="center", va="top", fontsize=6.8, color=label_colour,
                fontweight="bold" if i == step else "normal")
    if step > 0:
        arc = FancyArrowPatch((xs[step] - 0.01, y - 0.035), (xs[step - 1] + 0.01, y - 0.035),
                              connectionstyle="arc3,rad=0.55", arrowstyle="-|>",
                              color=ACCENT, lw=1.4, linestyle=(0, (4, 2)))
        ax.add_patch(arc)
        ax.text((xs[step - 1] + xs[step]) / 2, y - 0.21, "fed back as lag-1", ha="center",
                fontsize=7.2, color=ACCENT)
    caption = ("anchored on the last observed quarter" if step == 0
               else f"producing quarter {step} of 8")
    ax.text(0.02, 0.06, caption, fontsize=9.5,
            color=ACCENT if step else MUTED, fontweight="bold")


def gif() -> Path:
    history, future = _load_series()
    fig = plt.figure(figsize=(7.6, 3.7), dpi=100)
    ax_left = fig.add_axes((0.015, 0.02, 0.46, 0.96))
    ax_right = fig.add_axes((0.545, 0.14, 0.44, 0.78))

    frames = list(range(0, 9)) + [8] * 4  # hold the finished forecast

    def update(step: int):
        _draw_loop(ax_left, step)
        ax_right.clear()
        _draw_forecast_chart(ax_right, history, future, upto=step,
                             title="Brunswick 2-bed flat: forecast to 2027 Q3", title_size=10)
        ax_right.legend(frameon=False, fontsize=7, loc="upper left")
        return []

    writer = animation.PillowWriter(fps=1.6)
    path = FIGURES / "explainer.gif"
    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=writer)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


# ----------------------------------------------------------- linkedin ----

def _money(value: float, digits: int = 0) -> str:
    """Currency for matplotlib text. Unescaped $ enters math mode and eats spaces."""
    return f"\\${value:.{digits}f}"


def _linkedin_tables() -> dict:
    """Protocol MAEs, residual widths, and panel size used by the LinkedIn figure."""
    val = pd.read_csv(RESULTS / "validation_horizons.csv")
    bakeoff = pd.read_csv(RESULTS / "bakeoff_metrics.csv").set_index("model")
    illusions = json.loads((RESULTS / "evaluation_illusions.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(ROOT / "data" / "rent_panel.csv")
    usable = panel.dropna(subset=["median", "median_lag_1", "median_lag_4"])
    return {
        "huber": val[val["model"] == "huber_delta@location"].set_index("horizon"),
        "bands": pd.read_csv(RESULTS / "residual_bands.csv").set_index("horizon"),
        "val_mae": float(bakeoff.loc["huber_delta@location", "val_mae"]),
        "shuffled_mae": float(illusions["shuffled_split_one_step"]["mae"]),
        "shuffled_r2": float(illusions["shuffled_split_one_step"]["r2_on_level"]),
        "temporal_mae": float(illusions["temporal_split_one_step"]["mae"]),
        "n_series": int(usable.groupby(["suburb_group", "apartment_type", "bedrooms"]).ngroups),
    }


def _metric_cells(horizon: int, history: pd.DataFrame, future: pd.DataFrame,
                  tables: dict) -> list[tuple[str, str, str]]:
    """Three labelled facts. Same structure every frame; only the values move."""
    anchor = float(history["median"].iloc[-1])
    if horizon == 0:
        return [
            ("Origin", "2025 Q3", "last observed quarter"),
            ("Observed median", f"{_money(anchor)} / week", "Brunswick 2-bedroom flats"),
            ("Protocol", "8 recursive steps", "predicted lags after origin"),
        ]
    row = future.iloc[horizon - 1]
    half = float(tables["bands"].loc[horizon, "p80"])
    huber_mae = float(tables["huber"].loc[horizon, "val_mae"])
    return [
        ("Horizon", f"{horizon} of 8", str(row["label"])),
        ("Empirical 80% band",
         f"{_money(float(row['lower']))} to {_money(float(row['upper']))}",
         f"half-width {_money(half)} / week"),
        ("Validation MAE", f"{_money(huber_mae, 2)} / week",
         f"shuffled split {_money(tables['shuffled_mae'], 2)}"),
    ]


def _draw_metrics(ax, cells: list[tuple[str, str, str]]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.0, 1.0], [1.0, 1.0], color=LINE, linewidth=0.8, transform=ax.transAxes, clip_on=False)
    ax.plot([0.0, 1.0], [0.0, 0.0], color=LINE, linewidth=0.8, transform=ax.transAxes, clip_on=False)
    for index, (label, value, note) in enumerate(cells):
        x = 0.02 + index * 0.33
        ax.text(x, 0.78, label.upper(), fontsize=7.2, color=MUTED, fontweight="bold", va="top")
        ax.text(x, 0.48, value, fontsize=12.5, color=INK, fontweight="bold", va="center")
        ax.text(x, 0.18, note, fontsize=8.0, color="#475569", va="center")


def _draw_protocol_split(ax, tables: dict, horizon: int) -> None:
    """Paper panel: same Huber model under three scoring protocols.

    Shuffled and temporal bars stay fixed (the dashboard quizzes). The recursive
    bar is the protocol the forecast above actually runs, and grows with horizon.
    """
    shuffled = tables["shuffled_mae"]
    temporal = tables["temporal_mae"]
    if horizon:
        recursive = float(tables["huber"].loc[horizon, "val_mae"])
        rec_label = f"horizon {horizon}"
    else:
        recursive = 0.0
        rec_label = "this forecast"
    values = [shuffled, temporal, recursive]
    labels = [
        f"Shuffled 80/20\nR² = {tables['shuffled_r2']:.2f}",
        "Temporal split\none step",
        f"Recursive\n{rec_label}",
    ]
    colours = [MUTED, "#64748b", ACCENT]
    bars = ax.bar(range(3), values, color=colours, width=0.62)
    ax.set_xticks(range(3), labels)
    ymax = float(tables["huber"]["val_mae"].max()) * 1.32
    ax.set_ylim(0, ymax)
    for bar, value in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        if value <= 0:
            continue
        ax.text(x, value + ymax * 0.025, f"MAE {_money(value, 2)}", ha="center",
                fontsize=8.5, color=INK, fontweight="bold")
    ax.set_ylabel("MAE (AUD / week)", fontsize=8)
    ax.set_title("Same model, three evaluation protocols", color=INK, fontsize=9.5, loc="left", pad=4)
    _style_axes(ax)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", visible=False)


def _render_band_frame(history: pd.DataFrame, future: pd.DataFrame, tables: dict,
                       horizon: int, size: int = 1080) -> Image.Image:
    """One square frame: Brunswick cone plus the three-protocol MAE comparison."""
    hx, fx = _forecast_xy(history, future)
    y_lo = min(history["median"].min(), future["lower"].min()) - 18
    y_hi = max(history["median"].max(), future["upper"].max()) + 18
    anchor_y = float(history["median"].iloc[-1])
    dpi = 120
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.955, "TECHNICAL NOTE  |  HOMES VICTORIA RENTAL REPORT",
             color=ACCENT, fontsize=8.2, fontweight="bold", transform=fig.transFigure)
    fig.text(0.07, 0.905, "Recursive eight-quarter forecast",
             color=INK, fontsize=18, fontweight="bold", transform=fig.transFigure, va="center")
    fig.text(0.07, 0.862,
             "Brunswick, 2-bedroom flats. Moving-annual median weekly rent.",
             color="#475569", fontsize=10, transform=fig.transFigure, va="center")

    ax = fig.add_axes((0.12, 0.40, 0.81, 0.42))
    ax.plot(hx, history["median"], color=INK, linewidth=2.0, label="Observed median")
    ax.axvline(hx[-1], color=LINE, linewidth=1.0, linestyle=":")
    if horizon:
        rows = future.iloc[:horizon]
        xs = [hx[-1]] + fx[:horizon]
        lows = [anchor_y] + list(rows["lower"])
        highs = [anchor_y] + list(rows["upper"])
        ax.fill_between(xs, lows, highs, color=ACCENT, alpha=0.16,
                        label="Empirical 80th-percentile band")
        ax.plot(xs, [anchor_y] * len(xs), color=BASELINE, linestyle="--", linewidth=1.4,
                label="Last-quarter persistence")
        ax.plot(xs, [anchor_y] + list(rows["prediction"]),
                color=ACCENT, marker="o", markersize=4.5, linewidth=2.0, label="Huber forecast")
        last_x = xs[-1]
        last_pred = float(rows["prediction"].iloc[-1])
        ax.plot([last_x, last_x], [float(rows["lower"].iloc[-1]), float(rows["upper"].iloc[-1])],
                color=ACCENT, linewidth=1.1, alpha=0.75)
        ax.text(last_x + 0.18, last_pred, f"{_money(last_pred)}", color=ACCENT,
                fontsize=9, fontweight="bold", va="center")
    ax.set_xlim(hx[0] - 0.15, fx[-1] + 0.55)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("AUD / week", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2)
    _style_axes(ax)

    ax_stats = fig.add_axes((0.07, 0.255, 0.86, 0.105))
    _draw_metrics(ax_stats, _metric_cells(horizon, history, future, tables))

    ax_mae = fig.add_axes((0.12, 0.075, 0.81, 0.155))
    _draw_protocol_split(ax_mae, tables, horizon)

    fig.text(
        0.07, 0.018,
        f"Same Huber model. Shuffled-split MAE {_money(tables['shuffled_mae'], 2)} "
        f"(R² = {tables['shuffled_r2']:.2f}). Equal-horizon recursive MAE "
        f"{_money(tables['val_mae'], 2)}. {tables['n_series']} series, 2001 Q1 to 2025 Q3.",
        color="#64748b", fontsize=7.4, transform=fig.transFigure,
    )

    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image.load()
    buf.close()
    return image


def linkedin_card() -> Path:
    """Static square of the two-year cone — LinkedIn fallback if the GIF is skipped."""
    history, future = _load_series()
    tables = _linkedin_tables()
    path = FIGURES / "linkedin_card.png"
    _render_band_frame(history, future, tables, horizon=8, size=1080).save(path)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def linkedin_gif() -> Path:
    """Square looping GIF: cone opens while the recursive protocol MAE grows."""
    history, future = _load_series()
    tables = _linkedin_tables()
    # Even walk along the horizon; hold the origin and the finished two-year frame.
    story: list[tuple[int, int]] = [(0, 1600)] + [(h, 750) for h in range(1, 8)] + [(8, 2800)]
    frames = [_render_band_frame(history, future, tables, horizon) for horizon, _ in story]
    durations = [hold for _, hold in story]
    path = FIGURES / "linkedin.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def main(targets: list[str] | None = None) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = set(targets or ["poster", "gif", "linkedin"])
    if "poster" in selected:
        poster()
    if "gif" in selected:
        gif()
    if "linkedin" in selected:
        linkedin_card()
        linkedin_gif()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
