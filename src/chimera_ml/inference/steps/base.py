from typing import Protocol

from chimera_ml.inference.context import InferenceContext


class BaseInferenceStep(Protocol):
    """Minimal protocol for a sequential inference step."""

    def run(self, ctx: InferenceContext) -> InferenceContext: ...
