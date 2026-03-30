import torch

from chimera_ml.core.registry import Registry
from chimera_ml.training.builders import (
    build_from_registry,
    build_optimizer,
    build_train_config,
)


def test_build_from_registry_allow_none():
    reg = Registry("x")
    out = build_from_registry(reg, None, allow_none=True)
    assert out is None


def test_build_from_registry_uses_default_name():
    reg = Registry("x")

    @reg.register("demo")
    def _demo(v: int = 1):
        return {"v": v}

    out = build_from_registry(reg, None, default_name="demo")
    assert out == {"v": 1}


def test_build_from_registry_smart_inject_filters_unknown_args():
    reg = Registry("x")

    @reg.register("demo")
    def _demo(v: int = 1):
        return {"v": v}

    out = build_from_registry(
        reg,
        {"name": "demo", "params": {"v": 2}},
        inject={"unknown": 123},
        smart_inject=True,
    )
    assert out == {"v": 2}


def test_build_optimizer_default_is_adamw_optimizer():
    model = torch.nn.Linear(2, 1)
    opt = build_optimizer(None, model)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_train_config_reads_params_dict():
    cfg = build_train_config({"params": {"epochs": 3, "mixed_precision": True}})
    assert cfg.epochs == 3
    assert cfg.mixed_precision is True
