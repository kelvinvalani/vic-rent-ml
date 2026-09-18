import sys
from pathlib import Path
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_panel():
    return pd.DataFrame(
        {
            "suburb_group": ["Melbourne", "Melbourne", "Richmond", "Richmond"],
            "apartment_type": ["flat", "flat", "house", "house"],
            "bedrooms": [2, 2, 3, 3],
            "year": [2023, 2024, 2023, 2024],
            "quarter": [1, 2, 1, 2],
            "distance_to_cbd": [2.5, 2.5, 4.0, 4.0],
            "hospitals": [3, 3, 1, 1],
            "schools": [10, 10, 5, 5],
            "train_dist": [0.3, 0.3, 0.8, 0.8],
            "postcodes": ["3000", "3000", "3121", "3121"],
            "median_lag_1": [500.0, 520.0, 700.0, 720.0],
            "median_lag_4": [480.0, 490.0, 680.0, 690.0],
            "median": [520.0, 535.0, 720.0, 750.0],
        }
    )
