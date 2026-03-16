import torch

from chimera_ml.core.registry import SCHEDULERS


@SCHEDULERS.register("steplr_scheduler")
def steplr_scheduler(*, optimizer, **params):
    return torch.optim.lr_scheduler.StepLR(optimizer, **params)


@SCHEDULERS.register("cosineannealinglr_scheduler")
def cosineannealinglr_scheduler(*, optimizer, **params):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)


@SCHEDULERS.register("reduceonplateau_scheduler")
def reduceonplateau_scheduler(*, optimizer, **params):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
