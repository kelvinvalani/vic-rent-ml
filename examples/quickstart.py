"""Run the real selection rule on a tiny synthetic panel. No government data needed."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vic_rent_ml.ingest import attach_lags
from vic_rent_ml.training import train


def synthetic_panel() -> pd.DataFrame:
    rows = []
    for group, pc, base in (("Melbourne", "3000", 400), ("Richmond", "3121", 500)):
        for period in range(2015 * 4 + 1, 2025 * 4 + 4):
            rows.append(
                {
                    "suburb_group": group,
                    "postcodes": pc,
                    "apartment_type": "flat",
                    "bedrooms": 2,
                    "region": "Inner Melbourne",
                    "year": (period - 1) // 4,
                    "quarter": (period - 1) % 4 + 1,
                    "median": base + (period - 2015 * 4) * 2 + (period % 4),
                }
            )
    return attach_lags(pd.DataFrame(rows))


def main() -> None:
    out = Path("artifacts/quickstart")
    payload = train(
        synthetic_panel(),
        out,
        candidate_names=["ridge_delta@lags", "huber_delta@lags"],
    )
    print(f"Winner: {payload['recipe']}")
    print(f"Trained through: {payload['trained_through']}")
    print(payload["forecasts"].head(8).to_string(index=False))
    print(f"Wrote artifact to {out / 'production_clone.joblib'}")


if __name__ == "__main__":
    main()
