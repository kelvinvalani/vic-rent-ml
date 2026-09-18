# Fair temporal model selection for Victorian rental medians

**Author:** Kelvin Valani  
**Date:** 18 September 2026  
**Repository:** [github.com/kelvinvalani/vic-rent-ml](https://github.com/kelvinvalani/vic-rent-ml)  
**Units:** Australian dollars per week throughout

## Abstract

We compare 24 learned forecasting configurations and two persistence baselines
for Victorian suburb-group moving-annual median rents. Every candidate is scored
with the same recursive temporal protocol: expanding-window origins, eight-quarter
forecasts that feed predictions back into lag features, and a single validation
objective. There is no family preference and no minimum tree-over-linear hurdle.

Huber regression with postcode categories wins this experiment with validation
MAE **23.80**, compared with **25.94** for the best tested XGBoost configuration
and **25.98** for last-quarter persistence. The subsequent 2025 Q1–Q3 audit gives
MAE **14.87**, an **11.74%** reduction against persistence. The exact selected
pipeline is then refitted through 2025 Q3 and shipped as a schema-2 artifact that
anyone can load locally.

These findings concern aggregate rent medians, not individual property valuations,
and do not establish that linear models generally outperform XGBoost.

## 1. Target and source data

Homes Victoria publishes the Rental Report using Residential Tenancies Bond
Authority bond records. The target is its **moving-annual median weekly rent**,
indexed by suburb group, dwelling type and bedroom count. Consecutive quarters
therefore describe overlapping annual windows. This smooths the target, induces
serial dependence and makes recent-rent baselines particularly important.

This run uses the historical transformed CSV bundled in `data/`. A fresh official
workbook download is a different experiment; the existing extraction has not been
independently reconciled cell-by-cell with that workbook. This is a reproducible
re-analysis of the repository data, not a claim to use the newest published file.

After requiring observed rents and exact calendar lag-1 and lag-4 values, the
panel contains **82,025 rows**, **862 dwelling series**, and **146 suburb groups**,
from **2001 Q1 through 2025 Q3**. There are **142 observed representative postcode
values**. The source's `CBD-St Kilda Rd` group has **297 rows without a postcode**;
these retain an explicit missing category and can be addressed by group name.
The source supplies bedroom categories 1–4. Unsupported combinations are not
assigned invented rents.

Lag construction joins series by exact quarter offsets. A gap is not treated as
one quarter merely because two rows are adjacent. Duplicate series-quarter keys,
nonpositive rents and nonfinite required training values are rejected. Scoring
uses series observed at each origin whose future outcome is available; suppressed,
missing or newly appearing series do not contribute to those forecasts.

All errors receive equal row weight within a horizon. Lease-count weighting
could answer a different, transaction-weighted question, but is not used here.
Future lease counts are unavailable and excluded from model inputs.

## 2. Why the modelling choices look like this

### 2.1 Delta-innovation target

Learned models predict the quarterly change:

```text
delta(t) = median(t) - median(t-1)
predicted_median(t) = median(t-1) + predicted_delta(t)
```

Nominal medians trend. Predicting the level asks an estimator to reconstruct
that trend from calendar features; tree ensembles additionally cannot emit a
level above the highest training leaf, so they clip after a boom. Adding a
predicted change to the last observed median bounds that failure mode and makes
last-quarter persistence the nested model “predicted delta = 0”.

### 2.2 Features that will exist at serving time

The common numeric features are year, quarter, bedrooms, lag-1 rent and lag-4
rent. The `lags` feature variant adds dwelling type as a categorical feature;
`location` also adds representative postcode. Numeric imputation and scaling and
categorical one-hot encoding are fitted inside each fold. Unknown postcode
categories at evaluation time are handled by the fitted encoder.

Lease counts are omitted. They are known in the historical tables and unknown
for future quarters. Training with counts (or substituting a dummy such as 50
at inference) is train/serve skew: validation would not be the error a user
gets.

Location amenity statistics and macroeconomic columns are omitted from this
experiment. Retrospective amenity snapshots are not a vintage-safe origin
feature, and an eight-quarter forecast would otherwise need an invented RBA or
CPI path. Optional as-of macro ingestion can be added later; its value is not
established by the results reported here.

### 2.3 Recursive evaluation, not one-step or random split

A production forecast for horizon 5 cannot see the true lag-1 from horizon 4.
Scoring one-step-ahead errors therefore understates the quantity we ship.
Every candidate, including persistence, is rolled forward for eight quarters
with predicted medians written back into the lag state.

Random train/test splits leak because moving-annual windows overlap by nine
months and because lag features would be built from “future” rows. The pipeline
refuses a panel that does not extend past the test origin rather than falling
back to a shuffle.

## 3. Candidate models

The candidates are ordinary least squares, Ridge (α = 10), Lasso (α = 0.1),
Elastic Net (α = 0.1, l1_ratio = 0.5), Huber (ε = 1.35, max_iter = 2000),
Bayesian Ridge, histogram gradient boosting (200 iterations, 15 leaves,
L2 = 10), random forest and Extra Trees (150 trees, depth 12, min leaf 10),
gradient boosting (200 trees, depth 3, learning rate 0.05), and XGBoost with
depths 3 and 6 (300 trees, learning rate 0.05, min child weight 10, L2 = 10,
row/column subsample 0.8, `hist` tree method). Each learned estimator is tested
with both feature variants. Randomized estimators use seed 42. Complete
estimator settings are saved in `docs/results/candidate_parameters.json`.

This is a finite, specified candidate grid, not exhaustive hyperparameter
search. The same feature variants and scoring protocol apply to every learned
family. No family receives a deployment preference or a minimum improvement
requirement. Two baselines can also win: the last observed median, and the
same-quarter-last-year median. Both are evaluated recursively using the same
available history.

## 4. Temporal protocol

| Stage | Training/forecast origins | Scored targets | Purpose |
|---|---|---|---|
| Validation | Expanding history ending at Q4 of 2018, 2019, 2020, 2021 | Each origin's next 8 quarters, ending no later than 2023 Q4 | Choose candidate |
| Calibration | Expanding origins from 2022 Q1 through 2024 Q3 | Only targets in 2024, at horizons 1–8 | Fit error-band widths |
| Final audit | One origin at 2024 Q4 | Available 2025 Q1–Q3 outcomes | Audit frozen winner and baselines |
| Deployment refit | All labelled data through 2025 Q3 | Forecasts through 2027 Q3 | Serve the selected pipeline |

A forecast starts with observed history at its origin. Each predicted rent becomes
the next step's lag-1 and, four steps later, lag-4. No future actual rent enters
the feature construction of that forecast. If an exact older lag is unavailable
because of a historical gap, lag-4 falls back to the origin's observed median;
this rule is shared by all candidates and by deployed forecasts.

The selection score is the arithmetic mean of MAE over horizons 1–8. This prevents
shorter horizons from dominating simply because more outcomes are available.
The secondary score is the mean of per-horizon RMSE, **not** the square root of
pooled MSE. Exact metric ties use the candidate identifier for determinism. Test
metrics never enter this ranking. All 26 candidates have the same **26,651**
validation forecast/outcome pairs, and no convergence warnings were recorded.

After selection, each horizon's 80th percentile absolute calibration error becomes
the symmetric half-width around its prediction. The final audit measures coverage
without adjusting those widths. Only three test horizons are observable: this
experiment does **not** provide a final-test result for horizons 4–8.

The chronological partition prevents direct future-target leakage in forecasting.
It is not a full historical-vintage simulation: quarter-end reference dates are
used as availability dates. Publication delays, retrospective revisions and
overlap among moving-annual targets remain limitations.

## 5. Results

### 5.1 Validation selection

![Equal-horizon mean validation MAE for all 26 candidates. Huber with postcode features is lowest.](figures/validation_mae.png)

Best feature variant per family (full 26-row ranking is in `docs/results/bakeoff_metrics.csv`):

| Candidate | Features | Validation MAE | Validation RMSE |
|---|---|---:|---:|
| **Huber** | **location** | **23.80** | **37.22** |
| Elastic Net | lags / location tie | 24.14 | 37.45 |
| Lasso | lags / location tie | 24.17 | 37.40 |
| Bayesian Ridge | lags | 24.22 | 37.53 |
| Ridge | lags | 24.22 | 37.53 |
| Ordinary least squares | lags | 24.22 | 37.53 |
| Extra Trees | location | 25.65 | 39.82 |
| Random forest | lags | 25.74 | 40.35 |
| XGBoost, depth 6 | location | 25.94 | 40.85 |
| Histogram gradient boosting | lags | 25.94 | 40.68 |
| Last-quarter persistence | — | 25.98 | 39.44 |
| Gradient boosting | lags | 26.25 | 40.00 |
| XGBoost, depth 3 | lags | 26.32 | 40.31 |
| Seasonal persistence | — | 30.60 | 45.19 |

Huber's validation improvement over persistence is **8.40%**. Adding postcode
reduces Huber MAE only from **23.857** to **23.799**: a small descriptive difference,
not evidence of a statistically reliable location effect. The policy nevertheless
selects the lower score, exactly as it would for a similarly small tree advantage.

Results vary substantially by origin:

| Origin | Huber/location MAE | Persistence MAE | XGBoost depth-6/location MAE |
|---|---:|---:|---:|
| 2018 Q4 | 14.70 | 15.81 | 17.31 |
| 2019 Q4 | 22.95 | **21.25** | 24.66 |
| 2020 Q4 | 23.24 | 25.05 | 24.53 |
| 2021 Q4 | 34.27 | 41.77 | 37.21 |

The winner does not beat persistence at every origin. Strong persistence and
robust linear performance are consistent with a smooth target and noisy changes.
Huber's reduced sensitivity to outliers is a plausible explanation, not a
causal finding. Tree tuning and alternative representations may improve results.

![Per-horizon validation MAE, calibrated 80th-percentile band width, and 2025 audit MAE.](figures/horizon_errors.png)

### 5.2 Frozen-winner audit

| Model | Test MAE | Test RMSE | Forecast/outcome pairs |
|---|---:|---:|---:|
| **Huber/location** | **14.87** | **28.19** | 2,472 |
| Last-quarter persistence | 16.85 | 29.51 | 2,472 |
| Seasonal persistence | 34.30 | 45.17 | 2,472 |

These are equal-horizon averages over the **three available test horizons**,
so they are not directly comparable with the eight-horizon validation average.
The winner was frozen before this audit; other learned candidates were not
re-ranked on the test data.

| Horizon | Test pairs | MAE | Empirical half-width | Observed coverage |
|---|---:|---:|---:|---:|
| 1 quarter | 829 | 11.07 | 17.29 | 84.80% |
| 2 quarters | 822 | 14.82 | 33.19 | 90.51% |
| 3 quarters | 821 | 18.73 | 48.56 | 92.57% |
| 4 quarters | Not observed | — | 67.78 | — |
| 5 quarters | Not observed | — | 84.07 | — |
| 6 quarters | Not observed | — | 97.50 | — |
| 7 quarters | Not observed | — | 108.75 | — |
| 8 quarters | Not observed | — | 116.33 | — |

Coverage exceeds 80% on observed test horizons, especially later ones, suggesting
the 2024 calibration errors were conservative for this particular 2025 period.
That does not guarantee coverage in the next market regime.

Pooled slice MAE is **12.77** for flats and **16.90** for houses; it rises to
**26.50** for four-bedroom properties versus **9.94** for two-bedroom properties.
Armadale has MAE **63.64** on only 18 test pairs. Small suburb slices and correlated
observations warrant caution: average performance conceals substantial local
errors. No independent-sample significance tests or confidence claims are made.

## 6. Production artifact and reproducibility

`vic-rent-ml train` freezes the winner identifier, calibrates bands, audits the
test window against persistence, and refits the exact selected pipeline on all
labelled quarters. The schema-2 artifact includes the fitted estimator, origin
history, precomputed recursive forecasts, residual widths, winner, evaluation
plan, dependency versions and data/source digests.

Saved-selection reuse (`--reuse-selection`) fails if validation data, training
source or evaluation settings change. Serving reads artifact history instead of
fabricating amenity values or lease counts.

Postcodes may represent more than one source group. Such forecasts are aggregated
and flagged as imputed; an explicit group disambiguates them. Combinations without
supported history fail rather than inventing a rent.

The published run used Python **3.12.3**, NumPy **2.4.6**, pandas **3.0.6**,
scikit-learn **1.9.1**, XGBoost **3.4.1**, and joblib **1.6.0**. Reproduce:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-model.txt
pip install -e ".[dev]"
vic-rent-ml train --panel data/rent_panel.csv --out artifacts/production_clone.joblib --reuse-selection
python examples/predict.py
```

Compact results live in `docs/results/`. `manifest.json` records model artifact
SHA-256 `5839b40f1c04ae6646b178e66f21a5dc34742fb801459cb7c744afbe9afac56d`.
Only load trusted joblib artifacts with compatible dependencies.

## 7. Conclusions and next experiment

Removing any family override is necessary, but consistent recursive evaluation
and exact-winner deployment matter equally. In this specified experiment, Huber
wins on measured validation error. XGBoost is eligible under the identical rule
and will be the shipped model if it wins a future bake-off.

The next evaluation should obtain verified source vintages and publication dates,
expand the tree tuning budget using validation data only, and collect enough new
quarters for an independent eight-horizon audit. Forecasts remain estimates of
aggregate medians; their residual bands must not be described as the range
containing 80% of individual property rents.

## References and attribution

1. Homes Victoria, [Rental Report — Quarterly: Moving Annual Rents by Suburb](https://discover.data.vic.gov.au/dataset/rental-report-quarterly-moving-annual-rents-by-suburb).
   DataVic identifies the source license as Creative Commons Attribution 4.0
   International. Retain Homes Victoria attribution when redistributing its data.
2. Hyndman and Athanasopoulos, [Forecasting: Principles and Practice — Time series cross-validation](https://otexts.com/fpp3/tscv.html).
3. scikit-learn, [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
   and [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
4. Chen and Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785).
