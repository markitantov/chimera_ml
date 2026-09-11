from typing import Protocol

from chimera_ml.inference.context import InferenceContext


class BaseInferenceStep(Protocol):
    """Protocol implemented by every inference step.

    A step receives an InferenceContext, may read existing artifacts, and
    returns the context after adding or updating artifacts. Steps should keep
    configuration in the object instance and avoid hidden global state.
    """

    def run(self, ctx: InferenceContext) -> InferenceContext: ...
