# %% [markdown]
# # Reproducible rent bake-off
#
# This notebook calls the same `vic_rent_ml.training.train` entry point as the CLI.
# Candidate selection uses recursive validation MAE, then RMSE, then model name.
# Calibration and final testing occur after selection. No family override applies.

# %%
from pathlib import Path

import pandas as pd

from vic_rent_ml.training import train

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
PANEL = ROOT / "data" / "rent_panel.csv"
OUT = ROOT / "artifacts"

# %% [markdown]
# Rebuild the panel from the bundled historical CSV if you need to:
#
# ```
# vic-rent-ml ingest --transformed data/VIC-data-transformed.csv \
#   --regions data/suburb-category.csv --out data/rent_panel.csv
# ```
#
# Re-running ingest can change column layout relative to the published panel.
# Use `--reuse-selection` only against `data/rent_panel.csv` as shipped.

# %%
panel = pd.read_csv(PANEL)
summary = train(panel, OUT, reuse_selection=(OUT / "winner.json").exists())
print(summary["recipe"], summary["trained_through"])

# %%
pd.read_csv(OUT / "bakeoff_metrics.csv").sort_values(["val_mae", "val_rmse", "model"]).head(10)

# %%
pd.read_csv(OUT / "test_metrics.csv")
