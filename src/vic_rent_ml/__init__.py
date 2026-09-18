"""vic-rent-ml: Temporal selection and recursive Victorian rental-median forecasts."""

__version__ = "1.0.0"

from .recipe import (
    fit_production_model,
    make_production_pipeline,
    predict_expected_median,
    walk_forward_abs_errors,
    band_table_from_errors,
)
from .serving import calculate_bounds, predict_expected_median_from_features, predict_from_artifact
from .training import evaluate, train

__all__ = [
    "fit_production_model",
    "make_production_pipeline",
    "predict_expected_median",
    "walk_forward_abs_errors",
    "band_table_from_errors",
    "calculate_bounds",
    "predict_expected_median_from_features",
    "predict_from_artifact",
    "evaluate",
    "train",
]
