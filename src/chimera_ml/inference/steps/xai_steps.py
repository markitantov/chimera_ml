"""Inference orchestration for explanation results."""

from dataclasses import dataclass
from typing import Any

from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.inference.context import InferenceContext


@dataclass
class XAIStep:
    """Validate explanation produced by a model-owning step."""

    explanation_artifact: str = "explanation"
    output_artifact: str = "explanations"

    def run(self, ctx: InferenceContext) -> InferenceContext:
        explanation = ctx.get_artifact(self.explanation_artifact)
        if explanation is None:
            raise ValueError(
                f"Missing explanation artifact '{self.explanation_artifact}'. "
                "A model-owning inference step must run explain_batch first."
            )
        ctx.set_artifact(self.output_artifact, explanation)
        return ctx


@INFERENCE_STEPS.register("xai_step")
def xai_step(**params: Any) -> XAIStep:
    """Build the registered ``xai_step``."""
    return XAIStep(**params)
