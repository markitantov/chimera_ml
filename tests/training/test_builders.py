import torch

from chimera_ml.core.registry import COLLATES, LOGGERS, Registry
from chimera_ml.training.builders import (
    BuildContext,
    build_collate,
    build_from_registry,
    build_logger,
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


def test_build_from_registry_smart_inject_passes_context_when_supported():
    reg = Registry("x")

    @reg.register("demo")
    def _demo(context=None):
        return context.get("data.num_classes")

    context = BuildContext()
    context.set("data.num_classes", 3)

    out = build_from_registry(
        reg,
        {"name": "demo", "params": {}},
        inject={"context": context},
        smart_inject=True,
    )
    assert out == 3


def test_build_optimizer_default_is_adamw_optimizer():
    model = torch.nn.Linear(2, 1)
    opt = build_optimizer(None, model)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_train_config_reads_params_dict():
    cfg = build_train_config({"params": {"epochs": 3, "mixed_precision": True}})
    assert cfg.epochs == 3
    assert cfg.mixed_precision is True


def test_build_collate_passes_context_when_supported():
    key = "_test_context_collate_builder"

    if key not in COLLATES._items:

        @COLLATES.register(key)
        def _demo_collate(context=None):
            return {"num_classes": context.get("data.num_classes")}

    context = BuildContext()
    context.set("data.num_classes", 4)

    out = build_collate({"name": key, "params": {}}, context=context)
    assert out == {"num_classes": 4}


def test_build_logger_passes_context_when_supported():
    key = "_test_context_logger_builder"

    if key not in LOGGERS._items:

        @LOGGERS.register(key)
        def _demo_logger(context=None):
            return {"stage": context.stage}

    context = BuildContext(stage="train")
    out = build_logger({"name": key, "params": {}}, context=context)
    assert out == {"stage": "train"}
