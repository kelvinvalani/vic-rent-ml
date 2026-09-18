# Contributing

Issues and pull requests are welcome.

- Keep training, evaluation and serving on the same recursive forecast path.
  Do not add a random-split fallback or a family override that can replace a
  better tree with a linear model.
- Lease counts and other quantities unknown at forecast time must not enter
  model features.
- If you change `bakeoff.py`, `recipe.py`, `ingest.py`, `training.py` or
  `config.py`, `--reuse-selection` will correctly refuse the saved winner.
  Re-run selection and commit the new `docs/results/` tables with the change.
- Tests should fail when a forecast reads future actuals or when selection
  reuse is asked to ignore a changed panel.

```sh
pip install -e ".[dev]"
pytest -q tests
flake8 src tests --max-line-length=120
```
