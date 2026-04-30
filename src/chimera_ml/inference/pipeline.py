from chimera_ml.inference.context import InferenceContext
from chimera_ml.inference.steps.base import BaseInferenceStep


class InferencePipeline:
    """Sequential inference pipeline."""

    def __init__(self, steps: list[BaseInferenceStep], *, name: str = "inference_pipeline") -> None:
        self.steps = steps
        self.name = name

    def run(self, ctx: InferenceContext) -> InferenceContext:
        for step in self.steps:
            ctx = step.run(ctx)

        return ctx
