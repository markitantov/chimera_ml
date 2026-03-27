import torch

from chimera_ml.core.registry import OPTIMIZERS


@OPTIMIZERS.register("adamw_optimizer")
def adamw_optimizer(
    *, 
    model: torch.nn.Module, 
    lr: float = 1e-3, 
    weight_decay: float = 0.0, 
    **kwargs
):
    return torch.optim.AdamW(
        model.parameters(), 
        lr=float(lr), 
        weight_decay=float(weight_decay), 
        **kwargs
    )


@OPTIMIZERS.register("adam_optimizer")
def adam_optimizer(
    *, 
    model: torch.nn.Module, 
    lr: float = 1e-3, 
    weight_decay: float = 0.0, 
    **kwargs
):
    return torch.optim.Adam(
        model.parameters(), 
        lr=float(lr), 
        weight_decay=float(weight_decay), 
        **kwargs
    )


@OPTIMIZERS.register("sgd_optimizer")
def sgd_optimizer(
    *,
    model: torch.nn.Module,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    nesterov: bool = False,
    **kwargs,
):
    return torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
        momentum=float(momentum),
        nesterov=bool(nesterov),
        **kwargs,
    )