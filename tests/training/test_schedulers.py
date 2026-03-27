import torch

from chimera_ml.core.registry import SCHEDULERS
from chimera_ml.training.schedulers import (
    cosineannealinglr_scheduler,
    reduceonplateau_scheduler,
    steplr_scheduler,
)


def _optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=1e-2)


def test_steplr_scheduler_builds_with_expected_params():
    optimizer = _optimizer()
    scheduler = steplr_scheduler(optimizer=optimizer, step_size=2, gamma=0.5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 2
    assert scheduler.gamma == 0.5


def test_cosineannealinglr_scheduler_builds_with_expected_params():
    optimizer = _optimizer()
    scheduler = cosineannealinglr_scheduler(optimizer=optimizer, T_max=5, eta_min=1e-5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 5
    assert scheduler.eta_min == 1e-5


def test_reduceonplateau_scheduler_builds_and_steps():
    optimizer = _optimizer()
    scheduler = reduceonplateau_scheduler(optimizer=optimizer, mode="min", patience=1, factor=0.5)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    scheduler.step(1.0)
    scheduler.step(1.1)


def test_scheduler_factories_are_registered():
    for key in ("steplr_scheduler", "cosineannealinglr_scheduler", "reduceonplateau_scheduler"):
        factory = SCHEDULERS.get(key)
        assert callable(factory)
