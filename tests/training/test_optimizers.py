import torch

from chimera_ml.core.registry import OPTIMIZERS
from chimera_ml.training.optimizers import adam_optimizer, adamw_optimizer, sgd_optimizer


def _model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


def test_adamw_optimizer_builds_with_expected_hparams():
    model = _model()
    opt = adamw_optimizer(model=model, lr=5e-4, weight_decay=1e-2)
    assert isinstance(opt, torch.optim.AdamW)
    assert float(opt.param_groups[0]["lr"]) == 5e-4
    assert float(opt.param_groups[0]["weight_decay"]) == 1e-2


def test_adam_optimizer_builds_with_expected_hparams():
    model = _model()
    opt = adam_optimizer(model=model, lr=1e-3, weight_decay=0.0)
    assert isinstance(opt, torch.optim.Adam)
    assert float(opt.param_groups[0]["lr"]) == 1e-3
    assert float(opt.param_groups[0]["weight_decay"]) == 0.0


def test_sgd_optimizer_builds_with_momentum_and_nesterov():
    model = _model()
    opt = sgd_optimizer(model=model, lr=1e-2, momentum=0.9, nesterov=True, weight_decay=1e-4)
    assert isinstance(opt, torch.optim.SGD)
    assert float(opt.param_groups[0]["lr"]) == 1e-2
    assert float(opt.param_groups[0]["momentum"]) == 0.9
    assert bool(opt.param_groups[0]["nesterov"]) is True
    assert float(opt.param_groups[0]["weight_decay"]) == 1e-4


def test_optimizer_factories_are_registered():
    for key in ("adam_optimizer", "adamw_optimizer", "sgd_optimizer"):
        factory = OPTIMIZERS.get(key)
        assert callable(factory)
