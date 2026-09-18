Shipped files from the published bake-off:

- `production_clone.joblib` — schema-2 artifact (fitted Huber pipeline, origin history, recursive forecasts, residual bands)
- `manifest.json` — identity, metrics, dependency pins, SHA-256
- `winner.json` / `bakeoff_metrics.csv` — validation ranking used for `--reuse-selection`
- `residual_bands.*` / `test_*.csv` — calibration widths and frozen-winner audit

Row-level `validation_*.csv` dumps are gitignored; compact horizon/origin tables are in `docs/results/`.
