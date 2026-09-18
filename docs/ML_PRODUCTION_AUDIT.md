# Modelling audit

Decisions encoded in this repository, and why they exist.

| Finding | Resolution |
|---|---|
| A tree winner could be replaced by a linear estimator within 5% MAE. | Remove the family override. Select by validation MAE, then RMSE, then name. |
| Previous comparison used test errors for selection. | Separate rolling-origin validation, residual-band calibration, and a later audit. |
| Training could fit Huber even when another candidate won. | `train()` refits the selected identifier, including a persistence winner. |
| One-step evaluation understated recursive forecasting errors. | Score horizons 1–8 with predicted rents fed back into lags. Average horizons equally. |
| Positional lags could bridge missing quarters. | Join lag-1 and lag-4 by calendar period; reject duplicate series/quarter keys. |
| Serving invented location statistics and rent lags. | Serve artifact history and precomputed recursive forecasts; reject unsupported keys. |
| Random splitting would leak overlapping moving-annual windows. | Each fold trains only through its origin. No random-split fallback. |
| Old benchmark numbers lacked saved evidence. | Publish metrics, hashes, candidate parameters, horizon and slice diagnostics. |
| A stub regression check always succeeded. | `check-regression` reads saved audit MAE and fails if the file or threshold is missing. |

## Artifact contract (schema version 2)

The joblib payload stores the fitted pipeline (or winning baseline), origin
history, recursive forecasts, anchor quarter, max horizon, calibrated residual
bands, winner record, evaluation plan, dependency versions, and data/source
hashes. Inputs exclude future lease counts.

Optional amenity and macro columns may sit on the panel; this experiment's zoo
uses only calendar, bedrooms, lags, dwelling type, and optional postcode.

## Remaining limitations

- The bundled CSV has not been reconciled cell-by-cell with a newly downloaded
  government workbook. Hashes make the experiment reproducible, not certified.
- Quarter labels are reference dates, not publication timestamps.
- Small hyperparameter grid; no claim of globally optimal XGBoost tuning.
- Overlapping moving-annual windows mean row counts overstate independence.
- Residual bands are retrospective forecast-error quantiles, not property-level
  80% intervals.
- Shared postcodes return a median of matching groups and set `is_imputed`.
- Load joblib files only from trusted publishers with compatible dependencies.

Measured numbers: [research paper](RESEARCH_PAPER.md) and [docs/results](results/).
