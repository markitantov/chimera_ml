import pytest
import torch

import chimera_ml.models as fusion_models
from chimera_ml.core.registry import (
    CALLBACKS,
    COLLATES,
    DATAMODULES,
    LOGGERS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
    Registry,
)
from chimera_ml.training.builders import (
    BuildContext,
    build_callbacks,
    build_collate,
    build_datamodule,
    build_from_registry,
    build_logger,
    build_loss,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
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


def test_build_from_registry_smart_inject_filters_injected_args_for_kwargs_factory():
    reg = Registry("x")

    @reg.register("demo")
    def _demo(v: int = 1, **kwargs):
        return {"v": v, "kwargs": kwargs}

    context = BuildContext(stage="train")
    out = build_from_registry(
        reg,
        {"name": "demo", "params": {"v": 2}},
        inject={"context": context, "unknown": 123},
        smart_inject=True,
    )
    assert out == {"v": 2, "kwargs": {}}


def test_build_optimizer_default_is_adamw_optimizer():
    model = torch.nn.Linear(2, 1)
    opt = build_optimizer(None, model)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_train_config_reads_params_dict():
    cfg = build_train_config({"params": {"epochs": 3, "mixed_precision": True}})
    assert cfg.epochs == 3
    assert cfg.mixed_precision is True


def test_build_optimizer_filters_context_for_kwargs_factory():
    key = "_test_optimizer_builder_filters_context"

    if key not in OPTIMIZERS._items:

        @OPTIMIZERS.register(key)
        def _demo_optimizer(*, model, lr: float = 1e-3, **kwargs):
            return {"model": model, "lr": lr, "kwargs": kwargs}

    model = torch.nn.Linear(2, 1)
    context = BuildContext(stage="train")
    out = build_optimizer({"name": key, "params": {"lr": 1e-2}}, model, context=context)
    assert out["model"] is model
    assert out["lr"] == 1e-2
    assert out["kwargs"] == {}


def test_build_scheduler_filters_context_for_kwargs_factory():
    key = "_test_scheduler_builder_filters_context"

    if key not in SCHEDULERS._items:

        @SCHEDULERS.register(key)
        def _demo_scheduler(*, optimizer, gamma: float = 0.1, **params):
            return {"optimizer": optimizer, "gamma": gamma, "params": params}

    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters(), lr=1e-3)
    context = BuildContext(stage="train")
    out = build_scheduler({"name": key, "params": {"gamma": 0.5}}, optimizer, context=context)
    assert out["optimizer"] is optimizer
    assert out["gamma"] == 0.5
    assert out["params"] == {}


def test_build_loss_filters_context_for_kwargs_factory():
    key = "_test_loss_builder_filters_context"

    if key not in LOSSES._items:

        @LOSSES.register(key)
        def _demo_loss(**params):
            return params

    out = build_loss({"name": key, "params": {}}, context=BuildContext(stage="train"))
    assert out == {}


def test_build_metrics_filter_context_for_kwargs_factory():
    key = "_test_metric_builder_filters_context"

    if key not in METRICS._items:

        @METRICS.register(key)
        def _demo_metric(**params):
            return params

    out = build_metrics([{"name": key, "params": {}}], context=BuildContext(stage="train"))
    assert out == [{}]


def test_build_callbacks_filter_context_for_kwargs_factory():
    key = "_test_callback_builder_filters_context"

    if key not in CALLBACKS._items:

        @CALLBACKS.register(key)
        def _demo_callback(**params):
            return params

    out = build_callbacks([{"name": key, "params": {}}], context=BuildContext(stage="train"))
    assert out == [{}]


def test_build_collate_filters_context_for_kwargs_factory():
    key = "_test_collate_builder_filters_context"

    if key not in COLLATES._items:

        @COLLATES.register(key)
        def _demo_collate(**params):
            return params

    out = build_collate({"name": key, "params": {}}, context=BuildContext(stage="train"))
    assert out == {}


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


def test_build_datamodule_injects_configured_collate_into_loader():
    key = "_test_datamodule_collate_injection"

    if key not in DATAMODULES._items:

        @DATAMODULES.register(key)
        def _demo_datamodule(*, collate_fn):
            from torch.utils.data import DataLoader, Dataset

            class _Dataset(Dataset):
                def __len__(self):
                    return 1

                def __getitem__(self, index):
                    return index

            class _DataModule:
                def train_dataloader(self):
                    return DataLoader(_Dataset(), batch_size=1, collate_fn=collate_fn)

            return _DataModule()

    collate = build_collate({"name": "multimodal_sequence_collate", "params": {}})
    datamodule = build_datamodule({"name": key, "params": {}}, collate_fn=collate)

    assert datamodule.train_dataloader().collate_fn is collate


def test_build_datamodule_without_collate_preserves_factory_behavior():
    key = "_test_datamodule_without_collate_injection"

    if key not in DATAMODULES._items:

        @DATAMODULES.register(key)
        def _demo_datamodule(context=None):
            return {"context": context}

    context = BuildContext(stage="train")
    datamodule = build_datamodule({"name": key, "params": {}}, context=context)

    assert datamodule == {"context": context}


def test_build_datamodule_rejects_unsupported_configured_collate():
    key = "_test_datamodule_without_collate_support"

    if key not in DATAMODULES._items:

        @DATAMODULES.register(key)
        def _demo_datamodule():
            return object()

    with pytest.raises(TypeError, match="does not support configured collation"):
        build_datamodule({"name": key, "params": {}}, collate_fn=lambda _: None)


def test_build_datamodule_rejects_conflicting_collate_configuration():
    key = "_test_datamodule_collate_conflict"

    if key not in DATAMODULES._items:

        @DATAMODULES.register(key)
        def _demo_datamodule(*, collate_fn):
            return collate_fn

    with pytest.raises(ValueError, match="configured in both"):
        build_datamodule(
            {"name": key, "params": {"collate_fn": "legacy"}},
            collate_fn=lambda _: None,
        )


def test_build_collate_reports_unknown_registry_key():
    with pytest.raises(KeyError, match="unknown key"):
        build_collate({"name": "missing_collate", "params": {}})


def test_build_logger_passes_context_when_supported():
    key = "_test_context_logger_builder"

    if key not in LOGGERS._items:

        @LOGGERS.register(key)
        def _demo_logger(context=None):
            return {"stage": context.stage}

    context = BuildContext(stage="train")
    out = build_logger({"name": key, "params": {}}, context=context)
    assert out == {"stage": "train"}


@pytest.mark.parametrize(
    ("name", "params"),
    [
        (
            "tfs_fusion_model",
            {"input_dims": {"text": 4, "audio": 4, "video": 4}, "shared_dim": 4, "output_dim": 2},
        ),
        (
            "tfs_dcmmmoe_fusion_model",
            {"input_dims": {"text": 4, "audio": 4, "video": 4}, "shared_dim": 4, "output_dim": 2},
        ),
        (
            "raav_fusion_model",
            {
                "input_dims": {"frame": 4, "context": 4},
                "shared_dim": 4,
                "output_dim": 2,
                "frame_modalities": ("frame",),
                "context_modalities": ["context"],
            },
        ),
        (
            "tomar_fusion_model",
            {"input_dims": {"text": 4, "audio": 4}, "shared_dim": 4, "anchor_modality": "text", "output_dim": 2},
        ),
    ],
)
def test_build_model_constructs_feature_only_fusion_models(name, params):
    model = build_model({"name": name, "params": params})

    assert tuple(model.input_dims) == tuple(params["input_dims"])
    assert not hasattr(model, "encoders")
    assert model.prediction.out_features == params["output_dim"]
    assert callable(MODELS.get(name))
    assert hasattr(fusion_models, name)
