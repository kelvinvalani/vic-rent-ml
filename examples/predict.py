"""Load the shipped artifact and print a Brunswick 2-bed-flat forecast."""

from __future__ import annotations

from pathlib import Path

import joblib

from vic_rent_ml.serving import calculate_bounds, predict_from_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "production_clone.joblib"


def main() -> None:
    if not ARTIFACT.exists():
        raise SystemExit(
            f"Missing {ARTIFACT}. Train first:\n"
            "  vic-rent-ml train --panel data/rent_panel.csv --out artifacts/production_clone.joblib --reuse-selection"
        )
    payload = joblib.load(ARTIFACT)
    year, quarter = 2026, 3
    result = predict_from_artifact(payload, "3056", "flat", 2, year, quarter)
    lower, upper, width = calculate_bounds(
        result["predicted_rent"],
        horizon=result["horizon"],
        residual_bands=payload.get("residual_bands"),
    )
    print(f"Recipe:          {payload['recipe']}")
    print(f"Trained through: {payload['trained_through']}")
    print(f"Suburb group:    {result['suburb_group']}")
    print(f"Target:          {year} Q{quarter} (horizon {result['horizon']})")
    print(f"Expected median: ${result['predicted_rent']:.2f} / week")
    print(f"Empirical band:  ${lower:.2f} – ${upper:.2f}  (±${width:.2f})")
    print(f"Last observed:   ${result['lag_1']:.2f}  (lag-4 ${result['lag_4']:.2f})")


if __name__ == "__main__":
    main()
