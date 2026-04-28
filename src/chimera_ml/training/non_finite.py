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
    if out.preds is not None and not torch.isfinite(out.preds).all():
        raise FloatingPointError(
            f"Non-finite predictions detected at split='{split}', epoch={epoch}, global_step={global_step}. "
            f"{non_finite_debug_context(out=out, loss=loss, batch=batch)}"
        )

    if loss is not None and not torch.isfinite(loss).all():
        raise FloatingPointError(
            f"Non-finite loss detected at split='{split}', epoch={epoch}, global_step={global_step}. "
            f"{non_finite_debug_context(out=out, loss=loss, batch=batch)}"
        )


def assert_finite_gradients(
    *,
    model: torch.nn.Module,
    split: str,
    epoch: int,
    global_step: int,
    out: ModelOutput,
    loss: torch.Tensor,
    batch: Batch,
) -> None:
    """Fail fast on NaN/Inf gradients after backward/unscale."""
    bad_grad_names = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and not torch.isfinite(param.grad).all()
    ]
    if bad_grad_names:
        raise FloatingPointError(
            f"Non-finite gradients detected at split='{split}', epoch={epoch}, global_step={global_step}. "
            f"bad_grad_params={bad_grad_names}. "
            f"{non_finite_debug_context(out=out, loss=loss, batch=batch)}"
        )
