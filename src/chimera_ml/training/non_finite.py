import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


def tensor_debug_summary(tensor: torch.Tensor | None) -> str:
    """Return compact shape/dtype/value/finite information for debugging."""
    if tensor is None:
        return "none"

    finite = bool(torch.isfinite(tensor).all().item())
    summary = f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, finite={finite}"
    if tensor.ndim == 0:
        summary = f"{summary}, value={float(tensor.detach().item())}"

    return summary


def non_finite_debug_context(
    *,
    out: ModelOutput,
    loss: torch.Tensor | None,
    batch: Batch,
) -> str:
    """Return compact output + batch context for NaN/Inf diagnostics."""
    inputs = {name: tensor_debug_summary(value) for name, value in batch.inputs.items()}
    targets = tensor_debug_summary(batch.targets)
    preds = tensor_debug_summary(out.preds)
    loss_summary = tensor_debug_summary(loss)
    return f"preds={preds}, loss={loss_summary}, inputs={inputs}, targets={targets}"


def _non_finite_step_message(
    *,
    split: str,
    epoch: int,
    global_step: int,
    out: ModelOutput,
    loss: torch.Tensor | None,
    batch: Batch,
) -> str | None:
    """Return a fail-fast message for non-finite predictions/loss values."""
    if out.preds is not None and not torch.isfinite(out.preds).all():
        return (
            f"Non-finite predictions detected at split='{split}', epoch={epoch}, global_step={global_step}. "
            f"{non_finite_debug_context(out=out, loss=loss, batch=batch)}"
        )

    if loss is not None and not torch.isfinite(loss).all():
        return (
            f"Non-finite loss detected at split='{split}', epoch={epoch}, global_step={global_step}. "
            f"{non_finite_debug_context(out=out, loss=loss, batch=batch)}"
        )

    return None


def assert_finite_step(
    *,
    split: str,
    epoch: int,
    global_step: int,
    out: ModelOutput,
    loss: torch.Tensor | None,
    batch: Batch,
) -> None:
    """Fail fast on NaN/Inf values and include enough context to debug the source."""
    message = _non_finite_step_message(
        split=split,
        epoch=epoch,
        global_step=global_step,
        out=out,
        loss=loss,
        batch=batch,
    )
    if message is not None:
        raise FloatingPointError(message)
