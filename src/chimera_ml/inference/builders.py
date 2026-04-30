from collections.abc import Mapping
from typing import Any

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference.config import InferenceConfig
from chimera_ml.inference.pipeline import InferencePipeline
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
    """Build a sequential inference pipeline from an already loaded inference config."""
    steps: list[BaseInferenceStep] = []
    for step_cfg in config.steps:
        if not isinstance(step_cfg, dict):
            raise TypeError("Each inference step config must be a mapping.")

        steps.append(build_inference_step(step_cfg, inject=inject))

    return InferencePipeline(steps, name=config.pipeline_name)
