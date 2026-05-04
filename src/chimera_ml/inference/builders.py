import warnings
from collections.abc import Mapping
from typing import Any

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference.config import InferenceConfig
from chimera_ml.inference.pipeline import InferenceGraphNode, InferencePipeline
from chimera_ml.inference.steps.base import BaseInferenceStep
from chimera_ml.training.builders import build_from_registry


def build_inference_step(cfg: dict[str, Any], *, inject: Mapping[str, Any] | None = None) -> BaseInferenceStep:
    """Build a single inference step from the inference-steps registry."""
    step = build_from_registry(
        INFERENCE_STEPS,
        cfg,
        inject=inject,
        smart_inject=True,
    )

    if not hasattr(step, "run"):
        raise TypeError(f"Inference step '{cfg.get('name')}' must define a 'run(ctx)' method.")

    return step


def build_inference_pipeline(
    config: InferenceConfig,
    *,
    inject: Mapping[str, Any] | None = None,
) -> InferencePipeline:
    """Build an inference pipeline from an already loaded inference config."""
    nodes: list[InferenceGraphNode] = []
    previous_node_id: str | None = None
    parallel_mode = config.parallel
    has_after = False

    for index, step_cfg in enumerate(config.steps):
        if not isinstance(step_cfg, dict):
            raise TypeError("Each inference step config must be a mapping.")

        node_id = _resolve_node_id(step_cfg, index=index)
        explicit_after = _normalize_after(step_cfg.get("after"))
        has_after = has_after or bool(explicit_after)
        after = explicit_after if parallel_mode else (previous_node_id,) if previous_node_id is not None else ()

        nodes.append(
            InferenceGraphNode(
                node_id=node_id,
                step=build_inference_step(step_cfg, inject=inject),
                after=after,
            )
        )

        previous_node_id = node_id

    if parallel_mode and not has_after and len(nodes) > 1:
        warnings.warn(
            "Inference pipeline parallel mode is enabled but no step defines 'after'; "
            "all steps will start as independent roots.",
            stacklevel=2,
        )

    return InferencePipeline(nodes=nodes, name=config.pipeline_name, parallel_mode=parallel_mode)


def _resolve_node_id(cfg: dict[str, Any], *, index: int) -> str:
    value = cfg.get("id", cfg.get("name"))
    step_id = str(value).strip() if value is not None else ""
    if not step_id:
        raise ValueError(f"Inference step at index {index} must define a non-empty 'name'.")

    return step_id


def _normalize_after(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, list):
        raise TypeError("Inference step field 'after' must be a string or a list of strings.")

    items: list[str] = []
    for raw_item in raw_items:
        item = str(raw_item).strip()
        if not item:
            raise ValueError("Inference step field 'after' cannot contain empty values.")

        items.append(item)

    return tuple(items)
