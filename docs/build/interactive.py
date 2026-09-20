"""Build docs/index.html: a single self-contained interactive explainer for the model.

No network calls, no chart library — all data is embedded as JSON and drawn with inline
SVG, so the page can be opened from disk, served from GitHub Pages, or linked on LinkedIn.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
RESULTS = DOCS / "results"
OUT = DOCS / "index.html"
HISTORY_QUARTERS = 44

FAMILY_LABELS = {
    "naive_persist": ("Last-quarter persistence", "baseline"),
    "naive_seasonal": ("Same quarter last year", "baseline"),
    "lin_delta": ("Ordinary least squares", "linear"),
    "ridge_delta": ("Ridge", "linear"),
    "lasso_delta": ("Lasso", "linear"),
    "enet_delta": ("Elastic Net", "linear"),
    "bayes_delta": ("Bayesian Ridge", "linear"),
    "huber_delta": ("Huber (robust)", "robust"),
    "hgb_delta": ("Hist gradient boosting", "trees"),
    "rf_delta": ("Random forest", "trees"),
    "et_delta": ("Extra Trees", "trees"),
    "gbr_delta": ("Gradient boosting", "trees"),
    "xgb_d3_delta": ("XGBoost (depth 3)", "trees"),
    "xgb_d6_delta": ("XGBoost (depth 6)", "trees"),
}


def _label(model: str) -> tuple[str, str]:
    base, _, variant = model.partition("@")
    name, family = FAMILY_LABELS.get(base, (base, "trees"))
    return (f"{name} · {variant}" if variant else name), family


def _period_label(period: int) -> str:
    year, quarter = divmod(int(period) - 1, 4)
    return f"{year} Q{quarter + 1}"


def _bakeoff() -> list[dict]:
    frame = pd.read_csv(RESULTS / "bakeoff_metrics.csv").sort_values("val_mae")
    rows = []
    for record in frame.itertuples():
        label, family = _label(record.model)
        rows.append({"label": label, "family": family, "mae": round(record.val_mae, 2),
                     "rmse": round(record.val_rmse, 2), "winner": record.model == "huber_delta@location"})
    return rows


def _forecasts() -> tuple[list[dict], dict]:
    panel = pd.read_csv(ROOT / "data" / "rent_panel.csv")
    forecasts = pd.read_csv(RESULTS / "deployment_forecasts.csv")
    cutoff = int(panel["period"].max()) - HISTORY_QUARTERS
    panel = panel[panel["period"] > cutoff]
    keys = ["suburb_group", "apartment_type", "bedrooms"]
    history: dict[str, dict] = {}
    for key, group in panel.groupby(keys, sort=True):
        group = group.sort_values("period")
        history["|".join(str(part) for part in key)] = {
            "periods": [_period_label(period) for period in group["period"]],
            "values": [round(float(value), 1) for value in group["median"]],
        }
    series = []
    for key, group in forecasts.groupby(keys, sort=True):
        identifier = "|".join(str(part) for part in key)
        if identifier not in history:
            continue
        group = group.sort_values("horizon")
        series.append({
            "id": identifier,
            "group": key[0],
            "dwelling": key[1],
            "bedrooms": int(key[2]),
            "anchor": round(float(group["anchor_median"].iloc[0]), 1),
            "labels": list(group["label"]),
            "point": [round(float(value), 1) for value in group["prediction"]],
            "low": [round(float(value), 1) for value in group["lower"]],
            "high": [round(float(value), 1) for value in group["upper"]],
        })
    return series, history


def build_payload() -> dict:
    illusions = json.loads((RESULTS / "evaluation_illusions.json").read_text())
    uncertainty = json.loads((RESULTS / "audit_uncertainty.json").read_text())
    horizons = pd.read_csv(RESULTS / "test_horizons.csv")
    bands = pd.read_csv(RESULTS / "residual_bands.csv")
    bedrooms = pd.read_csv(RESULTS / "test_by_bedrooms.csv")
    dwelling = pd.read_csv(RESULTS / "test_by_apartment_type.csv")
    suburbs = pd.read_csv(RESULTS / "test_by_suburb_group.csv").sort_values("mae")
    series, history = _forecasts()
    return {
        "bakeoff": _bakeoff(),
        "protocols": [
            {"name": "Shuffled 80/20 split", "detail": "one step ahead, real previous quarter supplied",
             "mae": round(illusions["shuffled_split_one_step"]["mae"], 2),
             "extra": f"R² {illusions['shuffled_split_one_step']['r2_on_level']:.4f}",
             "verdict": "Looks superb, and is meaningless: overlapping annual windows and neighbouring "
                        "quarters of the same suburb sit on both sides of the split."},
            {"name": "Temporal split, one step", "detail": "trained on the past, but still handed the real lag",
             "mae": round(illusions["temporal_split_one_step"]["mae"], 2),
             "extra": f"R² {illusions['temporal_split_one_step']['r2_on_level']:.4f}",
             "verdict": "Honest about time, dishonest about deployment — at horizon 5 nobody knows the "
                        "previous quarter's rent yet."},
            {"name": "Recursive, 8 quarters", "detail": "the model eats its own forecasts, exactly as shipped",
             "mae": round(illusions["recursive_eight_horizon"]["mae"], 2),
             "extra": f"h=1 {illusions['recursive_eight_horizon']['mae_horizon_1']:.1f} → "
                      f"h=8 {illusions['recursive_eight_horizon']['mae_horizon_8']:.1f}",
             "verdict": "Three times worse than the shuffled number, and the only one that describes the "
                        "product. This is the number we publish."},
        ],
        "horizons": [
            {"horizon": int(row.horizon), "mae": round(row.mae, 2), "coverage": round(row.coverage * 100, 1),
             "width": round(row.width, 1), "pairs": int(row.n)}
            for row in horizons.itertuples()
        ],
        "bands": [{"horizon": int(row.horizon), "width": round(row.p80, 1)} for row in bands.itertuples()],
        "slices": {
            "bedrooms": [{"label": f"{int(row.bedrooms)} bed", "mae": round(row.mae, 2), "n": int(row.n)}
                         for row in bedrooms.itertuples()],
            "dwelling": [{"label": row.apartment_type, "mae": round(row.mae, 2), "n": int(row.n)}
                         for row in dwelling.itertuples()],
            "best": [{"label": row.suburb_group, "mae": round(row.mae, 2), "n": int(row.n)}
                     for row in suburbs.head(6).itertuples()],
            "worst": [{"label": row.suburb_group, "mae": round(row.mae, 2), "n": int(row.n)}
                      for row in suburbs.tail(6).iloc[::-1].itertuples()],
        },
        "uncertainty": uncertainty["windows"],
        "series": series,
        "history": history,
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How well can you really forecast Victorian rents?</title>
<meta property="og:title" content="How well can you really forecast Victorian rents?">
<meta property="og:description" content="26 models, one honest temporal protocol, 862 rental series.
Interactive walkthrough of the bake-off, the evaluation trap, and the shipped forecasts.">
<meta property="og:type" content="article">
<meta property="og:image" content="figures/social_card.png">
<meta name="twitter:card" content="summary_large_image">
<style>__STYLE__</style>
</head>
<body>
<main>
__BODY__
</main>
<script>const DATA = __DATA__;</script>
<script>__SCRIPT__</script>
</body>
</html>
"""

STYLE = """
:root {
  --ink:#0f2740; --accent:#c2410c; --teal:#0f766e; --violet:#7c3aed;
  --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --card:#ffffff;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  line-height:1.6; -webkit-font-smoothing:antialiased;
}
main { max-width:1060px; margin:0 auto; padding:2.5rem 1.2rem 5rem; }
h1 { font-size:clamp(2rem,4.6vw,3rem); line-height:1.1; letter-spacing:-0.025em; margin:0 0 .8rem; }
h2 { font-size:1.45rem; letter-spacing:-0.01em; margin:0 0 .3rem; }
p { margin:0 0 .9rem; }
.lede { font-size:1.12rem; color:#33475f; max-width:52rem; }
.kicker {
  text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700;
  color:var(--accent); margin-bottom:.7rem;
}
section { background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:1.7rem 1.6rem; margin-top:1.6rem; box-shadow:0 1px 2px rgba(15,39,64,.04); }
section > p { color:#3f5872; max-width:60rem; }
.hero { background:linear-gradient(135deg,#0f2740,#1d3f62 55%,#2a5580); color:#fff; border:0; }
.hero p { color:#cfe0f0; }
.hero .kicker { color:#f6b48a; }
.stats { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); margin-top:1.4rem; }
.stat { background:rgba(255,255,255,.09); border-radius:12px; padding:.9rem 1rem; }
.stat b { display:block; font-size:1.75rem; line-height:1.1; letter-spacing:-.02em; }
.stat span { font-size:.82rem; color:#bcd2e6; }
.cards { display:grid; gap:.9rem; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); margin:1.2rem 0; }
.card { text-align:left; cursor:pointer; background:#f1f5f9; border:1.5px solid transparent;
  border-radius:12px; padding:.9rem 1rem; font:inherit; color:inherit; transition:.15s; }
.card:hover { background:#e6edf5; }
.card.active { background:#fff; border-color:var(--accent); box-shadow:0 4px 16px rgba(194,65,12,.12); }
.card b { display:block; font-size:.98rem; }
.card small { color:var(--muted); display:block; margin-top:.15rem; line-height:1.4; }
.card .mae { font-size:1.6rem; font-weight:700; letter-spacing:-.02em; margin-top:.5rem; display:block; }
.verdict { border-left:3px solid var(--accent); padding:.1rem 0 .1rem .9rem; color:#3f5872; }
.controls { display:flex; flex-wrap:wrap; gap:.7rem; align-items:flex-end; margin:1rem 0 .4rem; }
label { display:block; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }
select { font:inherit; padding:.42rem .6rem; border:1px solid #cbd5e1; border-radius:8px;
  background:#fff; color:var(--ink); max-width:320px; }
.readout { display:flex; flex-wrap:wrap; gap:1.6rem; margin-top:.8rem; font-size:.9rem; color:var(--muted); }
.readout b { display:block; font-size:1.2rem; color:var(--ink); }
svg { width:100%; height:auto; display:block; }
.legend { display:flex; flex-wrap:wrap; gap:1.1rem; font-size:.8rem; color:var(--muted); margin-top:.6rem; }
.swatch { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:.35rem;
  vertical-align:-1px; }
.filters { display:flex; flex-wrap:wrap; gap:.5rem; margin:.9rem 0 1rem; }
.chip { font:inherit; font-size:.82rem; padding:.3rem .8rem; border-radius:999px; cursor:pointer;
  border:1px solid #cbd5e1; background:#fff; color:#33475f; }
.chip.active { background:var(--ink); border-color:var(--ink); color:#fff; }
.bars { display:flex; flex-direction:column; gap:.28rem; }
.bar { display:grid; grid-template-columns:minmax(120px,1.4fr) 5fr auto; gap:.6rem; align-items:center;
  font-size:.83rem; }
.bar .track { display:block; background:#eef2f7; border-radius:5px; overflow:hidden; height:16px; }
.bar .fill { display:block; height:100%; border-radius:5px; min-width:2px; }
.bar.win .name { font-weight:700; }
.grid2 { display:grid; gap:1.4rem; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); }
table { border-collapse:collapse; width:100%; font-size:.88rem; margin-top:.6rem; }
th, td { text-align:left; padding:.45rem .5rem; border-bottom:1px solid var(--line); }
th { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
ul.caveats { margin:.4rem 0 0; padding-left:1.1rem; color:#3f5872; }
ul.caveats li { margin-bottom:.4rem; }
footer { margin-top:1.8rem; font-size:.85rem; color:var(--muted); text-align:center; }
a { color:var(--accent); }
"""

BODY = """
<section class="hero">
  <div class="kicker">Victorian rental forecasting · open method, open numbers</div>
  <h1>How well can you <em>really</em> forecast suburb rents?</h1>
  <p class="lede">Twenty-six candidate models, 862 rental series, 146 suburb groups and one rule:
  every model is scored the way the product actually runs — forecasting eight quarters ahead while
  eating its own predictions. Baselines are allowed to win. This page is the short version; the
  <a href="vic-rent-ml-paper.pdf">full paper</a> has the maths.</p>
  <div class="stats">
    <div class="stat"><b>$14.87</b><span>per-week MAE on the frozen 2025 audit</span></div>
    <div class="stat"><b>11.7%</b><span>better than “rent stays the same”</span></div>
    <div class="stat"><b>__SERIES__</b><span>series still live in 2025 Q3, all forecast to 2027 Q3</span></div>
    <div class="stat"><b>0</b><span>rows of future data used to pick the model</span></div>
  </div>
</section>

<section>
  <div class="kicker">The trap</div>
  <h2>Same model. Same features. Three ways to score it.</h2>
  <p>These are not three models. They are the <b>same shipped model</b> sat three different exams.
  A lower number here means an easier exam, not a better forecaster — only the last one asks the
  question the product actually answers. Click a protocol.</p>
  <div class="cards" id="protocol-cards"></div>
  <p class="verdict" id="protocol-verdict"></p>
</section>

<section>
  <div class="kicker">The bake-off</div>
  <h2>Every candidate, one protocol, no favourites</h2>
  <p>Equal-horizon mean validation MAE over eight recursive quarters, in dollars per week — lower is
  better. The teal bar is last-quarter persistence &mdash; predicting no change at all &mdash; and it
  still beats six of the 24 learned configurations, including four boosting variants.</p>
  <div class="filters" id="bakeoff-filters"></div>
  <div class="bars" id="bakeoff-bars"></div>
</section>

<section>
  <div class="kicker">The product</div>
  <h2>What a shipped forecast looks like</h2>
  <p>Pick any series. The solid line is the observed moving-annual median; the shaded cone is the
  calibrated 80th-percentile error band, which widens from about ±$17 at one quarter to ±$116 at eight.</p>
  <div class="controls">
    <div><label for="pick-group">Suburb group</label><select id="pick-group"></select></div>
    <div><label for="pick-dwelling">Dwelling</label><select id="pick-dwelling"></select></div>
    <div><label for="pick-bedrooms">Bedrooms</label><select id="pick-bedrooms"></select></div>
  </div>
  <div id="forecast-chart"></div>
  <div class="readout" id="forecast-readout"></div>
  <div class="legend">
    <span><i class="swatch" style="background:#0f2740"></i>observed median</span>
    <span><i class="swatch" style="background:#c2410c"></i>forecast</span>
    <span><i class="swatch" style="background:#f2cdb8"></i>80% empirical band</span>
  </div>
</section>

<section>
  <div class="kicker">Honesty check</div>
  <h2>Error grows with distance — and the bands know it</h2>
  <div class="grid2">
    <div>
      <div id="horizon-chart"></div>
      <div class="legend">
        <span><i class="swatch" style="background:#c2410c"></i>audit MAE</span>
        <span><i class="swatch" style="background:#cbd5e1"></i>calibrated half-width</span>
      </div>
    </div>
    <div>
      <table>
        <thead><tr><th>Horizon</th><th class="num">MAE</th><th class="num">Band</th>
        <th class="num">Coverage</th></tr></thead>
        <tbody id="horizon-table"></tbody>
      </table>
      <p style="font-size:.85rem;margin-top:.7rem;color:#64748b">Only three horizons have real 2025
      outcomes. Horizons 4–8 ship with calibrated but unaudited bands — stated, not hidden.</p>
    </div>
  </div>
</section>

<section>
  <div class="kicker">Where it struggles</div>
  <h2>Averages hide the hard slices</h2>
  <div class="grid2">
    <div><h3 style="font-size:.95rem;margin:0 0 .4rem">By bedrooms and dwelling type</h3>
      <div class="bars" id="slice-bars"></div></div>
    <div><h3 style="font-size:.95rem;margin:0 0 .4rem">Best and worst suburb groups (18 pairs each)</h3>
      <div class="bars" id="suburb-bars"></div></div>
  </div>
</section>

<section>
  <div class="kicker">Is the win real?</div>
  <h2>Resampling whole suburb groups, 2,000 times</h2>
  <p>Rows inside a suburb group are not independent, so the bootstrap resamples entire groups rather
  than rows. The gap against persistence survives in both windows, and no draw reverses the ranking.</p>
  <table>
    <thead><tr><th>Window</th><th class="num">Pairs</th><th class="num">Model</th><th class="num">Persistence</th>
    <th class="num">Gap</th><th class="num">95% interval</th><th class="num">Series won</th></tr></thead>
    <tbody id="bootstrap-table"></tbody>
  </table>
</section>

<section>
  <div class="kicker">Read this before quoting the number</div>
  <h2>What this model does not do</h2>
  <ul class="caveats">
    <li>It forecasts <b>group medians</b>, not individual properties. The bands are not the range of
    individual rents.</li>
    <li>It does not prove linear models beat gradient boosting in general — only in this grid, on this
    target, under this protocol.</li>
    <li>It explains nothing causally. Postcode and bedroom effects are associations inside a pipeline.</li>
    <li>It has no audited accuracy beyond three quarters ahead.</li>
    <li>It assumes the measurement process is stable; a change in bond-lodgement coverage or the
    moving-annual definition would break it silently.</li>
  </ul>
</section>

<footer>
  Data: Homes Victoria Rental Report (RTBA bond lodgements), CC BY 4.0 · Method and code:
  <a href="https://github.com/kelvinvalani/vic-rent-ml">github.com/kelvinvalani/vic-rent-ml</a> ·
  <a href="vic-rent-ml-paper.pdf">Read the paper (PDF)</a>
</footer>
"""

SCRIPT = """
const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const FAMILY = { baseline:"#0f766e", linear:"#2563eb", robust:"#c2410c", trees:"#7c3aed" };

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- protocol cards ---------- */
function renderProtocols() {
  const wrap = $("protocol-cards");
  DATA.protocols.forEach((protocol, index) => {
    const card = document.createElement("button");
    card.className = "card" + (index === DATA.protocols.length - 1 ? " active" : "");
    card.innerHTML = `<b>${protocol.name}</b><small>${protocol.detail}</small>
      <span class="mae" style="color:${index === 2 ? "#c2410c" : "#64748b"}">$${protocol.mae.toFixed(2)}</span>
      <small>MAE per week · ${protocol.extra}</small>`;
    card.onclick = () => {
      [...wrap.children].forEach((child) => child.classList.remove("active"));
      card.classList.add("active");
      $("protocol-verdict").textContent = protocol.verdict;
    };
    wrap.appendChild(card);
  });
  $("protocol-verdict").textContent = DATA.protocols[DATA.protocols.length - 1].verdict;
}

/* ---------- bake-off ---------- */
let bakeoffFilter = "all";
function renderBakeoff() {
  const filters = $("bakeoff-filters");
  const options = [["all", "All 26"], ["baseline", "Baselines"], ["linear", "Linear"],
    ["robust", "Robust linear"], ["trees", "Trees & boosting"]];
  filters.innerHTML = "";
  options.forEach(([key, label]) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (key === bakeoffFilter ? " active" : "");
    chip.textContent = label;
    chip.onclick = () => { bakeoffFilter = key; renderBakeoff(); };
    filters.appendChild(chip);
  });
  const max = Math.max(...DATA.bakeoff.map((row) => row.mae));
  const persistence = DATA.bakeoff.find((row) => row.label.startsWith("Last-quarter")).mae;
  const bars = $("bakeoff-bars");
  bars.innerHTML = "";
  DATA.bakeoff
    .filter((row) => bakeoffFilter === "all" || row.family === bakeoffFilter)
    .forEach((row) => {
      const line = document.createElement("div");
      line.className = "bar" + (row.winner ? " win" : "");
      const marker = (persistence / max) * 100;
      line.innerHTML = `<span class="name">${row.label}</span>
        <span class="track" style="background:linear-gradient(90deg,#eef2f7 ${marker}%,#e2e8f0 ${marker}%)">
          <span class="fill" style="width:${(row.mae / max) * 100}%;background:${FAMILY[row.family]}"></span>
        </span>
        <span style="font-variant-numeric:tabular-nums">$${row.mae.toFixed(2)}</span>`;
      bars.appendChild(line);
    });
}

/* ---------- forecast explorer ---------- */
function renderExplorer() {
  const groups = [...new Set(DATA.series.map((s) => s.group))].sort();
  const groupSelect = $("pick-group");
  groups.forEach((group) => groupSelect.append(new Option(group, group)));
  groupSelect.value = groups.includes("Brunswick") ? "Brunswick" : groups[0];
  groupSelect.onchange = () => syncDwelling();
  $("pick-dwelling").onchange = () => syncBedrooms();
  $("pick-bedrooms").onchange = () => drawForecast();
  syncDwelling();
}

function syncDwelling() {
  const matches = DATA.series.filter((s) => s.group === $("pick-group").value);
  const select = $("pick-dwelling");
  const previous = select.value;
  select.innerHTML = "";
  [...new Set(matches.map((s) => s.dwelling))].sort()
    .forEach((value) => select.append(new Option(value, value)));
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
  syncBedrooms();
}

function syncBedrooms() {
  const matches = DATA.series.filter(
    (s) => s.group === $("pick-group").value && s.dwelling === $("pick-dwelling").value);
  const select = $("pick-bedrooms");
  const previous = select.value;
  select.innerHTML = "";
  [...new Set(matches.map((s) => s.bedrooms))].sort()
    .forEach((value) => select.append(new Option(value, value)));
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
  drawForecast();
}

function drawForecast() {
  const series = DATA.series.find((s) => s.group === $("pick-group").value
    && s.dwelling === $("pick-dwelling").value && s.bedrooms === Number($("pick-bedrooms").value));
  const host = $("forecast-chart");
  host.innerHTML = "";
  if (!series) return;
  const history = DATA.history[series.id];
  const W = 900, H = 340, padL = 56, padR = 18, padT = 18, padB = 34;
  const labels = history.periods.concat(series.labels);
  const n = labels.length;
  const values = history.values.concat(series.high, series.low);
  const lo = Math.min(...values), hi = Math.max(...values);
  const pad = (hi - lo) * 0.12 + 5;
  const yMin = lo - pad, yMax = hi + pad;
  const x = (i) => padL + (i * (W - padL - padR)) / (n - 1);
  const y = (v) => padT + (H - padT - padB) * (1 - (v - yMin) / (yMax - yMin));
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const value = yMin + ((yMax - yMin) * i) / ticks;
    svg.appendChild(el("line", { x1: padL, x2: W - padR, y1: y(value), y2: y(value),
      stroke: "#eef2f7", "stroke-width": 1 }));
    svg.appendChild(el("text", { x: padL - 8, y: y(value) + 4, "text-anchor": "end",
      "font-size": 11, fill: "#94a3b8" }, `$${Math.round(value)}`));
  }
  const step = Math.ceil(n / 9);
  labels.forEach((label, i) => {
    if (i % step) return;
    svg.appendChild(el("text", { x: x(i), y: H - 10, "text-anchor": "middle", "font-size": 11,
      fill: "#94a3b8" }, label));
  });

  const base = history.values.length - 1;
  const bandTop = series.high.map((v, i) => `${x(base + 1 + i)},${y(v)}`);
  const bandBottom = series.low.map((v, i) => `${x(base + 1 + i)},${y(v)}`).reverse();
  const anchor = `${x(base)},${y(history.values[base])}`;
  svg.appendChild(el("polygon", { points: [anchor, ...bandTop, ...bandBottom].join(" "),
    fill: "#c2410c", "fill-opacity": 0.16 }));
  svg.appendChild(el("polyline", { fill: "none", stroke: "#0f2740", "stroke-width": 2.2,
    points: history.values.map((v, i) => `${x(i)},${y(v)}`).join(" ") }));
  svg.appendChild(el("polyline", { fill: "none", stroke: "#c2410c", "stroke-width": 2.4,
    points: [anchor, ...series.point.map((v, i) => `${x(base + 1 + i)},${y(v)}`)].join(" ") }));
  series.point.forEach((v, i) => svg.appendChild(el("circle", { cx: x(base + 1 + i), cy: y(v), r: 3.2,
    fill: "#c2410c" })));
  host.appendChild(svg);

  const last = series.point.length - 1;
  $("forecast-readout").innerHTML = `
    <div><b>$${series.anchor.toFixed(0)}</b>latest observed median (2025 Q3)</div>
    <div><b>$${series.point[0].toFixed(0)}</b>${series.labels[0]} forecast
      (±$${((series.high[0] - series.low[0]) / 2).toFixed(0)})</div>
    <div><b>$${series.point[last].toFixed(0)}</b>${series.labels[last]} forecast
      (±$${((series.high[last] - series.low[last]) / 2).toFixed(0)})</div>
    <div><b>${(((series.point[last] / series.anchor) - 1) * 100).toFixed(1)}%</b>implied two-year change</div>`;
}

/* ---------- horizons ---------- */
function renderHorizons() {
  const W = 480, H = 300, padL = 44, padB = 34, padT = 14, padR = 10;
  const widths = DATA.bands.map((row) => row.width);
  const max = Math.max(...widths) * 1.08;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  const band = (W - padL - padR) / DATA.bands.length;
  const y = (v) => padT + (H - padT - padB) * (1 - v / max);
  DATA.bands.forEach((row, i) => {
    const audit = DATA.horizons.find((h) => h.horizon === row.horizon);
    const left = padL + i * band + band * 0.16;
    const width = band * 0.68;
    svg.appendChild(el("rect", { x: left, y: y(row.width), width, height: y(0) - y(row.width),
      fill: "#cbd5e1", rx: 3 }));
    if (audit) {
      svg.appendChild(el("rect", { x: left + width * 0.22, y: y(audit.mae), width: width * 0.56,
        height: y(0) - y(audit.mae), fill: "#c2410c", rx: 3 }));
    }
    svg.appendChild(el("text", { x: left + width / 2, y: H - 12, "text-anchor": "middle",
      "font-size": 11, fill: "#64748b" }, `h${row.horizon}`));
    svg.appendChild(el("text", { x: left + width / 2, y: y(row.width) - 5, "text-anchor": "middle",
      "font-size": 10, fill: "#64748b" }, `$${Math.round(row.width)}`));
  });
  svg.appendChild(el("line", { x1: padL, x2: W - padR, y1: y(0), y2: y(0), stroke: "#cbd5e1" }));
  $("horizon-chart").appendChild(svg);
  $("horizon-table").innerHTML = DATA.bands.map((row) => {
    const audit = DATA.horizons.find((h) => h.horizon === row.horizon);
    return `<tr><td>${row.horizon} quarter${row.horizon > 1 ? "s" : ""}</td>
      <td class="num">${audit ? "$" + audit.mae.toFixed(2) : "—"}</td>
      <td class="num">±$${row.width.toFixed(0)}</td>
      <td class="num">${audit ? audit.coverage.toFixed(1) + "%" : "not observed"}</td></tr>`;
  }).join("");
}

/* ---------- slices ---------- */
function barList(host, rows, colour) {
  const max = Math.max(...rows.map((row) => row.mae));
  host.innerHTML = rows.map((row) => `<div class="bar">
    <span class="name">${row.label}</span>
    <span class="track"><span class="fill" style="width:${(row.mae / max) * 100}%;
      background:${typeof colour === "function" ? colour(row) : colour}"></span></span>
    <span style="font-variant-numeric:tabular-nums">$${row.mae.toFixed(1)}</span></div>`).join("");
}

function renderSlices() {
  barList($("slice-bars"), DATA.slices.bedrooms.concat(DATA.slices.dwelling),
    (row) => (row.label.includes("bed") ? "#c2410c" : "#0f2740"));
  const suburbs = DATA.slices.best.concat(DATA.slices.worst);
  barList($("suburb-bars"), suburbs, (row) => (DATA.slices.best.includes(row) ? "#0f766e" : "#c2410c"));
}

/* ---------- bootstrap ---------- */
function renderBootstrap() {
  $("bootstrap-table").innerHTML = DATA.uncertainty.map((window) => `<tr>
    <td>${window.window}</td>
    <td class="num">${window.pairs.toLocaleString()}</td>
    <td class="num">$${window.winner_mae.toFixed(2)}</td>
    <td class="num">$${window.baseline_mae.toFixed(2)}</td>
    <td class="num">$${window.bootstrap.observed_mae_gap.toFixed(2)}</td>
    <td class="num">[${window.bootstrap.ci95_low.toFixed(2)}, ${window.bootstrap.ci95_high.toFixed(2)}]</td>
    <td class="num">${(window.series_win_rate * 100).toFixed(1)}%</td></tr>`).join("");
}

renderProtocols();
renderBakeoff();
renderExplorer();
renderHorizons();
renderSlices();
renderBootstrap();
"""


def main() -> int:
    data = build_payload()
    payload = json.dumps(data, separators=(",", ":"))
    html = (TEMPLATE
            .replace("__STYLE__", STYLE)
            .replace("__BODY__", BODY.replace("__SERIES__", str(len(data["series"]))))
            .replace("__SCRIPT__", SCRIPT)
            .replace("__DATA__", payload))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
