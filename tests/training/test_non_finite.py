import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput
from chimera_ml.training.non_finite import assert_finite_step, non_finite_debug_context, tensor_debug_summary


def _batch() -> Batch:
    return Batch(
        inputs={"x": torch.tensor([[1.0], [2.0]], dtype=torch.float32)},
        targets=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        masks=None,
        meta=None,
    )


def test_tensor_debug_summary_handles_none_and_scalar():
    assert tensor_debug_summary(None) == "none"
    assert "shape=(), dtype=torch.float32, finite=True, value=1.5" in tensor_debug_summary(torch.tensor(1.5))


def test_non_finite_debug_context_includes_inputs_targets_preds_and_loss():
    batch = _batch()
    out = ModelOutput(preds=torch.tensor([[0.0], [1.0]], dtype=torch.float32))
    loss = torch.tensor(0.25, dtype=torch.float32)

    summary = non_finite_debug_context(out=out, loss=loss, batch=batch)

    assert "preds=shape=(2, 1), dtype=torch.float32, finite=True" in summary
    assert "loss=shape=(), dtype=torch.float32, finite=True, value=0.25" in summary
    assert "inputs={'x': 'shape=(2, 1), dtype=torch.float32, finite=True'}" in summary
    assert "targets=shape=(2, 1), dtype=torch.float32, finite=True" in summary


def test_assert_finite_step_raises_for_non_finite_predictions():
    batch = _batch()
    out = ModelOutput(preds=torch.full((2, 1), float("nan"), dtype=torch.float32))

    with pytest.raises(FloatingPointError, match="Non-finite predictions detected"):
        assert_finite_step(split="val", epoch=1, global_step=0, out=out, loss=None, batch=batch)


def test_assert_finite_step_raises_for_non_finite_loss():
    batch = _batch()
    out = ModelOutput(preds=torch.tensor([[0.0], [1.0]], dtype=torch.float32))
    loss = torch.tensor(float("inf"), dtype=torch.float32)

    with pytest.raises(FloatingPointError, match="Non-finite loss detected"):
        assert_finite_step(split="val", epoch=1, global_step=0, out=out, loss=loss, batch=batch)
