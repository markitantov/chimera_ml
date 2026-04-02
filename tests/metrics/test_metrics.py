import numbers

import numpy as np
import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.metrics.confusion_matrix_metric import ConfusionMatrixMetric
from chimera_ml.metrics.prf_metric import PRFMetric
from chimera_ml.metrics.regression_metric import (
    MAEMetric,
    MSEMetric,
    R2Metric,
    RMSEMetric,
)

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _assert_close(actual, expected) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    np.testing.assert_allclose(actual_arr, expected_arr, rtol=1e-12, atol=1e-12, equal_nan=True)


def _logits_from_pred_indices(y_pred: np.ndarray, num_classes: int) -> torch.Tensor:
    logits = torch.full((y_pred.shape[0], num_classes), -5.0, dtype=torch.float32)
    idx = torch.arange(y_pred.shape[0], dtype=torch.long)
    logits[idx, torch.tensor(y_pred, dtype=torch.long)] = 5.0
    return logits


@pytest.mark.parametrize(
    ("scenario", "y_true", "y_pred"),
    [
        ("single_class", np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])),
        ("multi_class", np.array([0, 0, 1, 1]), np.array([0, 2, 2, 2])),
    ],
)
@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize("zero_division", [0, 1])
def test_classification_metrics_match_sklearn(
    scenario: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str,
    zero_division: int,
):
    metric = PRFMetric(average=average, zero_division=zero_division)
    metric.reset()

    num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    logits = _logits_from_pred_indices(y_pred=y_pred, num_classes=num_classes)
    batch = Batch(inputs={}, targets=torch.tensor(y_true, dtype=torch.long))
    out = ModelOutput(preds=logits)
    metric.update(out, batch)

    m = metric.compute()
    prefix = average

    expected_precision = sklearn_metrics.precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    expected_recall = sklearn_metrics.recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    expected_f1 = sklearn_metrics.f1_score(y_true, y_pred, average=average, zero_division=zero_division)

    _assert_close(m[f"{prefix}_precision"], expected_precision)
    _assert_close(m[f"{prefix}_recall"], expected_recall)
    _assert_close(m[f"{prefix}_f1"], expected_f1)


@pytest.mark.parametrize("normalize", [None, "true", "pred", "all"])
def test_confusion_matrix_metric_matches_sklearn(normalize: str | None):
    metric = ConfusionMatrixMetric(normalize=normalize)
    out = ModelOutput(
        preds=torch.tensor(
            [
                [3.0, 0.1, 0.0],
                [0.1, 2.0, 0.0],
                [0.2, 0.1, 1.8],
                [0.2, 0.1, 1.7],
            ],
            dtype=torch.float32,
        )
    )
    batch = Batch(inputs={}, targets=torch.tensor([0, 1, 2, 1], dtype=torch.long))
    metric.update(out, batch)

    m = metric.compute()
    cm = metric.value()

    assert cm is not None
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 2, 2])
    expected_cm = sklearn_metrics.confusion_matrix(y_true, y_pred, normalize=normalize)
    _assert_close(cm, expected_cm)

    expected_acc = float(np.trace(expected_cm) / max(np.sum(expected_cm), 1.0))
    _assert_close(m["cm_acc"], expected_acc)


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
@pytest.mark.parametrize(
    ("cls", "key", "sk_fn"),
    [
        (MAEMetric, "mae", sklearn_metrics.mean_absolute_error),
        (MSEMetric, "mse", sklearn_metrics.mean_squared_error),
        (RMSEMetric, "rmse", sklearn_metrics.root_mean_squared_error),
    ],
)
def test_regression_metrics_match_sklearn(
    multioutput: str,
    cls,
    key: str,
    sk_fn,
):
    preds = torch.tensor(
        [[0.2, 2.4, 0.8], [2.8, 3.9, 4.5], [4.5, 5.1, 5.8], [6.0, 7.3, 8.1]],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[0.0, 2.0, 1.0], [3.0, 4.0, 4.0], [5.0, 5.0, 6.0], [6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    metric = cls(multioutput=multioutput)
    metric.reset()
    metric.update(out, batch)
    m = metric.compute()

    expected = sk_fn(targets.numpy(), preds.numpy(), multioutput=multioutput)
    _assert_close(m[key], expected)


@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", "variance_weighted"])
def test_r2_metric_matches_sklearn(multioutput: str):
    preds = torch.tensor(
        [[0.2, 2.4, 0.8], [2.8, 3.9, 4.5], [4.5, 5.1, 5.8], [6.0, 7.3, 8.1]],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[0.0, 2.0, 1.0], [3.0, 4.0, 4.0], [5.0, 5.0, 6.0], [6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    metric = R2Metric(multioutput=multioutput)
    metric.reset()
    metric.update(out, batch)
    m = metric.compute()

    expected = sklearn_metrics.r2_score(targets.numpy(), preds.numpy(), multioutput=multioutput)
    _assert_close(m["r2"], expected)


def test_metrics_compute_empty_after_reset_returns_empty_dict():
    prf = PRFMetric(average="macro")
    prf.reset()
    assert prf.compute() == {}

    cm = ConfusionMatrixMetric()
    cm.reset()
    assert cm.compute() == {}
    assert cm.value() is None


def test_confusion_matrix_reset_clears_previous_matrix_value():
    metric = ConfusionMatrixMetric()
    out = ModelOutput(preds=torch.tensor([[3.0, 0.1], [0.1, 2.0]]))
    batch = Batch(inputs={}, targets=torch.tensor([0, 1]))

    metric.update(out, batch)
    metric.compute()
    assert metric.value() is not None

    metric.reset()
    assert metric.value() is None
    assert metric.compute() == {}


def test_regression_metric_shape_mismatch_raises_value_error():
    metric = MAEMetric()
    out = ModelOutput(preds=torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    batch = Batch(inputs={}, targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))

    metric.reset()
    with pytest.raises(ValueError, match="must match targets shape"):
        metric.update(out, batch)


def test_regression_invalid_multioutput_matches_sklearn_constraints():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    metric = MAEMetric(multioutput="variance_weighted")
    metric.reset()
    metric.update(out, batch)

    with pytest.raises(ValueError, match="Allowed values for this metric"):
        metric.compute()


def test_regression_metrics_return_numeric_scalars():
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=targets)

    for metric in (MAEMetric(), MSEMetric(), RMSEMetric(), R2Metric()):
        metric.reset()
        metric.update(out, batch)
        vals = metric.compute()
        assert len(vals) == 1
        assert isinstance(next(iter(vals.values())), numbers.Real)
