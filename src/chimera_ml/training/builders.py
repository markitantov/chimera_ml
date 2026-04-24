import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch

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
)
from chimera_ml.training.config import TrainConfig


class ContextDescribable(Protocol):
    """Optional protocol for components that enrich BuildContext during registration."""

    def describe_context(self, context: "BuildContext") -> None: ...


@dataclass
class BuildContext:
    """Per-run build context shared across datamodule/model/loss/etc construction."""

    config: Any | None = None
    stage: str | None = None
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, path: str, default: Any = None) -> Any:
        if not path:
            return default

        node: Any = self.values
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default

            node = node[part]

        return node

    def set(self, path: str, value: Any) -> None:
        parts = [part.strip() for part in path.split(".") if part.strip()]
        if not parts:
            raise ValueError("Context path cannot be empty.")

        node = self.values
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child

            node = child

        node[parts[-1]] = value

    def register(self, component: Any) -> Any:
        """Let a built component enrich the shared context."""
        if hasattr(component, "describe_context"):
            describable = cast(ContextDescribable, component)
            describable.describe_context(self)

        return component

    def register_many(self, components: list[Any]) -> list[Any]:
        """Let a list of built components enrich the shared context."""
        for component in components:
            self.register(component)

        return components


def build_from_registry(
    registry: Any,
    cfg: dict[str, Any] | None,
    *,
    default_name: str | None = None,
    allow_none: bool = False,
    normalize_name: bool = True,
    params_key: str = "params",
    name_key: str = "name",
    inject: Mapping[str, Any] | None = None,
    inject_overrides: bool = True,
    smart_inject: bool = False,
) -> Any:
    """
    Universal builder for Registry-based factories.

    Expected config format:
      cfg = {"name": "<key>", "params": {...}}

    Args:
        registry:
            Registry-like object with `get(name)` that returns a callable factory.
        cfg:
            Config dict or None.
        default_name:
            Name used if cfg is None or cfg[name_key] is missing. If None and cfg is None:
              - return None if allow_none=True
              - else raise
        allow_none:
            If True, returns None when cfg is None (and default_name is None) or cfg has empty name.
        normalize_name:
            If True, uses lower() on name.
        params_key/name_key:
            Keys used in cfg.
        inject:
            Runtime dependencies (e.g. model_params, optimizer) merged into params.
        inject_overrides:
            If True, inject values override cfg params on conflicts.
        smart_inject:
            If True, only pass injected keys that the factory accepts (by signature introspection).
            Useful when you want to inject `param` universally, but some factories don't accept it.

    Returns:
        Built object (factory(**kwargs)) or None (if allow_none and cfg/default implies None).
    """
    # 1) Resolve cfg / name
    if cfg is None:
        if default_name is None:
            if allow_none:
                return None
            raise ValueError("cfg is None and default_name is None (set allow_none=True to return None).")
        cfg = {name_key: default_name, params_key: {}}

    name = cfg.get(name_key, default_name)
    if name is None or (isinstance(name, str) and name.strip() == ""):
        if allow_none:
            return None
        raise ValueError(f"Missing '{name_key}' in cfg and default_name is None.")

    name = str(name)
    if normalize_name:
        name = name.lower()

    # 2) Resolve params
    params = cfg.get(params_key, {}) or {}
    if not isinstance(params, dict):
        raise TypeError(f"Expected cfg['{params_key}'] to be a dict, got: {type(params)}")

    # 3) Resolve factory
    factory = registry.get(name)
    if not callable(factory):
        raise TypeError(f"Registry '{registry}' returned non-callable for name='{name}': {type(factory)}")

    # 4) Merge inject into params
    kwargs = dict(params)
    if inject:
        if inject_overrides:
            kwargs.update(inject)
        else:
            for k, v in inject.items():
                kwargs.setdefault(k, v)

    # 5) Optionally filter injected keys by factory signature
    if smart_inject and inject:
        try:
            sig = inspect.signature(factory)
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                allowed = set(sig.parameters.keys())
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except (TypeError, ValueError):
            # Can't inspect -> fall back to passing everything
            pass

    return factory(**kwargs)


def build_loss(cfg: dict[str, Any], *, context: BuildContext | None = None) -> Any:
    """Build loss function from the losses registry."""
    inject = {"context": context} if context is not None else None
    return build_from_registry(LOSSES, cfg, inject=inject, smart_inject=True)


def build_metrics(cfg_list: list[dict[str, Any]], *, context: BuildContext | None = None) -> list[Any]:
    """Build all metrics from metric configs."""
    inject = {"context": context} if context is not None else None
    return [build_from_registry(METRICS, mcfg, inject=inject, smart_inject=True) for mcfg in cfg_list]


def build_datamodule(cfg: dict[str, Any], *, context: BuildContext | None = None) -> object:
    """Build a datamodule from registry config."""
    inject = {"context": context} if context is not None else None
    return build_from_registry(DATAMODULES, cfg, inject=inject, smart_inject=True)


def build_model(cfg: dict[str, Any], *, context: BuildContext | None = None) -> Any:
    """Build model from the models registry."""
    inject = {"context": context} if context is not None else None
    return build_from_registry(MODELS, cfg, inject=inject, smart_inject=True)


def build_optimizer(
    cfg: dict[str, Any] | None,
    model: torch.nn.Module,
    *,
    context: BuildContext | None = None,
) -> torch.optim.Optimizer:
    """Build optimizer, defaulting to AdamW when config is not provided."""
    inject = {"model": model}
    if context is not None:
        inject["context"] = context

    return build_from_registry(
        OPTIMIZERS,
        cfg,
        default_name="adamw_optimizer",
        inject=inject,
        inject_overrides=True,
        smart_inject=True,
    )


def build_scheduler(
    cfg: dict[str, Any] | None,
    optimizer: torch.optim.Optimizer,
    *,
    context: BuildContext | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build scheduler from config or return `None`."""
    inject = {"optimizer": optimizer}
    if context is not None:
        inject["context"] = context

    return build_from_registry(
        SCHEDULERS,
        cfg,
        allow_none=True,
        inject=inject,
        inject_overrides=True,
        smart_inject=True,
    )


def build_callbacks(cfg_list: list[dict[str, Any]] | None, *, context: BuildContext | None = None) -> list[Any]:
    """Build callbacks from callback configs."""
    if not cfg_list:
        return []

    inject = {"context": context} if context is not None else None
    return [build_from_registry(CALLBACKS, ccfg, inject=inject, smart_inject=True) for ccfg in cfg_list]


def build_collate(cfg: dict[str, Any] | None, *, context: BuildContext | None = None) -> Any:
    """Build collate callable, defaulting to `masking_collate`."""
    inject = {"context": context} if context is not None else None
    return build_from_registry(
        COLLATES,
        cfg,
        default_name="masking_collate",
        inject=inject,
        smart_inject=True,
    )


def build_logger(
    cfg: dict[str, Any] | None,
    *,
    inject: dict[str, Any] | None = None,
    context: BuildContext | None = None,
) -> Any | None:
    """Build optional logger with runtime injection."""
    merged_inject = dict(inject or {})
    if context is not None:
        merged_inject["context"] = context

    return build_from_registry(
        LOGGERS,
        cfg,
        allow_none=True,
        inject=merged_inject or None,
        inject_overrides=True,
        smart_inject=True,
    )


def build_train_config(cfg: dict[str, Any]) -> TrainConfig:
    """Build `TrainConfig` from yaml section."""
    params = cfg.get("params", {}) or {}
    return TrainConfig(**params)
