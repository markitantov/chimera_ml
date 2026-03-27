import numbers

import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.sklearn_classification import SklearnPRFMetric
from chimera_ml.metrics.sklearn_confusion_matrix import SklearnConfusionMatrixMetric
from chimera_ml.metrics.sklearn_regression import (
    SklearnMAEMetric,
    SklearnMSEMetric,
    SklearnR2Metric,
    SklearnRMSEMetric,
)


def test_classification_metrics_compute():
    metric = SklearnPRFMetric(average="macro")
    metric.reset()
    out = ModelOutput(preds=torch.tensor([[3.0, 0.1], [0.2, 2.0], [2.0, 0.1]]))
    batch = Batch(inputs={}, targets=torch.tensor([0, 1, 0]))
    metric.update(out, batch)
    m = metric.compute()
    assert "macro_precision" in m
    assert "macro_recall" in m
    assert "macro_f1" in m


def test_confusion_matrix_metric_compute_and_value():
    metric = SklearnConfusionMatrixMetric()
    out = ModelOutput(preds=torch.tensor([[2.0, 0.1], [0.1, 3.0], [2.0, 0.1]]))
    batch = Batch(inputs={}, targets=torch.tensor([0, 1, 0]))
    metric.update(out, batch)
    m = metric.compute()
    assert "cm_acc" in m
    cm = metric.value()
    assert cm is not None
    assert cm.shape[0] == cm.shape[1]


def test_regression_metrics_compute():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    for cls, key in [
        (SklearnMAEMetric, "mae"),
        (SklearnMSEMetric, "mse"),
        (SklearnRMSEMetric, "rmse"),
        (SklearnR2Metric, "r2"),
    ]:
        metric = cls()
        metric.reset()
        metric.update(out, batch)
        m = metric.compute()
        assert key in m


def test_metrics_compute_empty_after_reset_returns_empty_dict():
    prf = SklearnPRFMetric(average="macro")
    prf.reset()
    assert prf.compute() == {}

    cm = SklearnConfusionMatrixMetric()
    cm.reset()
    assert cm.compute() == {}
    assert cm.value() is None


def test_confusion_matrix_reset_clears_previous_matrix_value():
    metric = SklearnConfusionMatrixMetric()
    out = ModelOutput(preds=torch.tensor([[3.0, 0.1], [0.1, 2.0]]))
    batch = Batch(inputs={}, targets=torch.tensor([0, 1]))

    metric.update(out, batch)
    metric.compute()
    assert metric.value() is not None

    metric.reset()
    assert metric.value() is None
    assert metric.compute() == {}


def test_regression_metric_shape_mismatch_raises_value_error():
    metric = SklearnMAEMetric()
    out = ModelOutput(preds=torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    batch = Batch(inputs={}, targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))

    metric.reset()
    with pytest.raises(ValueError, match="must match targets shape"):
        metric.update(out, batch)


def test_regression_metrics_return_numeric_scalars():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    for metric in (SklearnMAEMetric(), SklearnMSEMetric(), SklearnRMSEMetric(), SklearnR2Metric()):
        metric.reset()
        metric.update(out, batch)
        vals = metric.compute()
        assert len(vals) == 1
        assert isinstance(next(iter(vals.values())), numbers.Real)
