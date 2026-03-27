import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.losses.ccc import CCCLoss
from chimera_ml.losses.classification import CrossEntropyLoss
from chimera_ml.losses.focal import FocalLoss
from chimera_ml.losses.multilabel import BCEWithLogitsLoss
from chimera_ml.losses.regression import MAELoss, MSELoss


def test_regression_losses_return_scalar():
    out = ModelOutput(preds=torch.tensor([[1.0], [2.0]]))
    batch = Batch(inputs={}, targets=torch.tensor([[1.5], [1.5]]))
    assert MSELoss()(out, batch).ndim == 0
    assert MAELoss()(out, batch).ndim == 0
    assert CCCLoss()(out, batch).ndim == 0


def test_classification_losses_return_scalar():
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    targets = torch.tensor([0, 1])
    out = ModelOutput(preds=logits)
    batch = Batch(inputs={}, targets=targets)

    assert CrossEntropyLoss()(out, batch).ndim == 0
    assert FocalLoss(gamma=2.0)(out, batch).ndim == 0


def test_multilabel_bce_with_logits_loss():
    logits = torch.tensor([[0.2, -0.3], [1.0, -1.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    out = ModelOutput(preds=logits)
    batch = Batch(inputs={}, targets=targets)
    assert BCEWithLogitsLoss()(out, batch).ndim == 0


def test_ccc_loss_is_close_to_zero_for_identical_tensors():
    preds = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.float32)
    out = ModelOutput(preds=preds)
    batch = Batch(inputs={}, targets=preds.clone())
    loss = CCCLoss()(out, batch)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_ccc_loss_shape_mismatch_raises_value_error():
    out = ModelOutput(preds=torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    batch = Batch(inputs={}, targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="must match targets shape"):
        CCCLoss()(out, batch)


def test_focal_loss_reduction_none_returns_per_sample_tensor():
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [2.0, -0.1]])
    targets = torch.tensor([0, 1, 0])
    out = ModelOutput(preds=logits)
    batch = Batch(inputs={}, targets=targets)

    loss = FocalLoss(reduction="none")(out, batch)
    assert loss.ndim == 1
    assert loss.shape[0] == targets.shape[0]


def test_focal_loss_unknown_reduction_raises_value_error():
    logits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    targets = torch.tensor([0, 1])
    out = ModelOutput(preds=logits)
    batch = Batch(inputs={}, targets=targets)
    with pytest.raises(ValueError, match="Unknown reduction"):
        FocalLoss(reduction="median")(out, batch)


def _make_mo():
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.2, 2.0, -0.5]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    return ModelOutput(preds=logits), Batch(inputs={}, targets=targets)


def test_focal_loss_sum_reduction_and_scalar_alpha():
    out, batch = _make_mo()
    loss = FocalLoss(alpha=0.5, reduction="sum")(out, batch)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_focal_loss_vector_alpha_and_label_smoothing_changes_value():
    out, batch = _make_mo()
    base = FocalLoss(alpha=[1.0, 2.0, 3.0], reduction="mean", label_smoothing=0.0)(out, batch)
    smooth = FocalLoss(alpha=[1.0, 2.0, 3.0], reduction="mean", label_smoothing=0.2)(out, batch)
    assert base.ndim == 0
    assert smooth.ndim == 0
    assert not torch.isclose(base, smooth)


def test_focal_loss_unknown_reduction_raises():
    out, batch = _make_mo()
    with pytest.raises(ValueError, match="Unknown reduction"):
        FocalLoss(reduction="bad")(out, batch)
