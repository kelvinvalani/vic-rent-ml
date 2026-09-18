"""Configuration settings and feature schemas for vic-rent-ml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FeatureConfig:
    # Numeric features for production pipeline (ZERO train/serve skew: count is excluded!)
    numeric_features: List[str] = field(
        default_factory=lambda: [
            "quarter",
            "year",
            "distanceToCbd",
            "hospitals",
            "schools",
            "trainDist",
            "bedrooms",
            "median_lag_1",
            "median_lag_4",
        ]
    )
    # Categorical features
    categorical_features: List[str] = field(
        default_factory=lambda: ["apartment_type", "postcodes"]
    )
    # Macroeconomic lagged indicators
    macro_lags: List[str] = field(
        default_factory=lambda: [
            "cpi_index_lag1",
            "cpi_index_lag4",
            "cpi_qoq_lag1",
            "cash_rate_lag1",
            "cash_rate_lag4",
        ]
    )


@dataclass
class ModelConfig:
    train_max_year: int = 2023
    val_year: int = 2024
    test_min_year: int = 2025
    walk_forward_horizons: int = 8
    residual_band_percentile: int = 80
    huber_max_iter: int = 400


@dataclass
class ProjectPaths:
    root: Path = field(default_factory=lambda: Path.cwd())
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    artifacts: Path = field(init=False)

    def __post_init__(self):
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.artifacts = self.root / "artifacts"
