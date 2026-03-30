"""Metrics package.

Only sklearn-based metrics are exposed and registered by default.
Importing this package ensures registry decorators are executed.
"""

from chimera_ml.metrics.base import BaseMetric
from chimera_ml.metrics.sklearn_classification import (
    prf_macro_metric,
    prf_micro_metric,
    prf_weighted_metric,
)
from chimera_ml.metrics.sklearn_confusion_matrix import confusion_matrix_metric
from chimera_ml.metrics.sklearn_regression import mae_metric, mse_metric, r2_metric, rmse_metric

__all__ = [
    "BaseMetric",
    "confusion_matrix_metric",
    "mae_metric",
    "mse_metric",
    "prf_macro_metric",
    "prf_micro_metric",
    "prf_weighted_metric",
    "r2_metric",
    "rmse_metric",
]
