# How the model works

This is the short companion to the [research paper](RESEARCH_PAPER.md). It explains the modelling choices in the order a new contributor usually meets them.

## The series

Each row is one **suburb group × dwelling type × bedrooms × quarter**. The label is Homes Victoria's **moving-annual median weekly rent**: the median of new bonds in the 12 months ending that quarter.

That definition matters:

- Consecutive quarters overlap by nine months, so the series is smooth and strongly autocorrelated.
- Last quarter's median is already a strong forecast. Any learned model has to beat **persistence**, not a random-split R².
- The number is a market level for a group, not a property valuation.

## The prediction problem

At origin quarter *T* we observe the panel through *T*. We want quarters *T+1 … T+8*.

A one-step model that is handed the true lag at every future quarter understates error. Serving a two-year forecast has to use **its own earlier predictions** as lag-1 (and four steps later as lag-4). Training and evaluation do the same thing. That is `forecast_panel` in `vic_rent_ml.recipe`.

## Delta, not level

Learned candidates predict

```text
delta(t)  = median(t) - median(t-1)
median̂(t) = median(t-1) + deltâ(t)
```

Reasons:

- Weekly rents have a unit-root flavour. Predicting the level asks a model to invent the local trend every time.
- Tree models cannot predict a rent higher than any training leaf. After a boom, level-trees clip. Delta-trees (and delta-linear models) add a change to the last observed median.
- Persistence is then just “predicted delta = 0”, so it sits in the same bake-off.

Seasonal persistence is the one level baseline: it copies the value from four quarters ago.

## Features that exist at origin time

Every learned model sees some of:

| Feature | Role |
|---|---|
| `year`, `quarter` | Calendar / seasonality |
| `bedrooms` | Dwelling size |
| `apartment_type` | `flat` or `house` |
| `median_lag_1` | Last quarter's median (observed, then predicted) |
| `median_lag_4` | Same quarter last year |
| `postcodes` | Representative postcode, `location` variants only |

Lease **count is not a feature**. It is known historically and unknown in the future. Using it would make validation look better than production.

Amenity columns and CPI / cash-rate columns can exist on the bundled panel. This experiment **does not feed them to the bake-off**. Retrospective amenities are not a vintage-safe origin feature, and a future RBA path would have to be invented. The package can carry those columns; the winning recipe ignores them.

Lags are joined by **calendar period**, not by “previous row”. A missing quarter is a gap, not a shifted neighbour.

## The bake-off

`build_model_zoo` builds 24 learned pipelines (linear, regularized linear, Huber, Bayesian Ridge, histogram / random / extra / gradient boosting, XGBoost depth 3 and 6) × (`lags` or `location` features), plus two baselines.

Shared rules:

1. Expanding-window origins at 2018 Q4, 2019 Q4, 2020 Q4, 2021 Q4.
2. Each origin forecasts eight quarters, scoring only targets through 2023 Q4.
3. Score = mean of the eight per-horizon MAEs (so h=1 does not dominate). Tie-break: mean per-horizon RMSE, then candidate name.
4. **No family preference.** If XGBoost wins, it ships.

2024 residuals calibrate the 80th-percentile absolute-error band at each horizon. 2025 Q1–Q3 is a frozen-winner audit. Test numbers never choose the model.

After that, the exact winner is refit on **all** labelled rows through 2025 Q3 and stored with history, precomputed recursive forecasts, bands, hashes, and dependency versions (schema version 2).

## Serving

`predict_from_artifact` does not rebuild features by guessing CBD distance or a dummy lease count. It looks up the suburb group in the artifact's origin history and returns the precomputed recursive forecast for that horizon. Unknown postcode × dwelling combinations are rejected rather than imputed silently.

A postcode that maps to several groups returns their median forecast and sets `is_imputed`. Pass `suburb_group` to disambiguate.

## What “good” means here

On this dataset a robust linear model beat a lightly tuned tree grid. That is a result for **this** smooth overlapping median, this grid, and this protocol. It is not a claim that Huber is generally better than XGBoost. Persistence is within about $2/week of the winner. The next bake-off can replace the winner without changing the serving contract.
