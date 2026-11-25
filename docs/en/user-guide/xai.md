# Research-aligned XAI

'xai_step' wraps an inference step that exposes an optional callback:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None

The wrapped step remains an ordinary prediction step. The wrapper installs the callback for the duration of 'run', invokes 'explain_batch' for every prediction batch, restores the previous callback in 'finally', and publishes the collected result as the 'xai' artifact. The existing 'predictions' artifact is unchanged.

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          ablation_policy: zero
          modalities: [x]

The nested step is built through 'INFERENCE_STEPS'. To make a custom prediction step compatible, add an optional callback field and call it after the model forward:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = field(default=None, repr=False)

    output = model(batch)
    if self.on_prediction is not None:
        self.on_prediction(model, batch, output)

A minimal registered custom step that can be constructed from the YAML above is:

    from collections.abc import Callable
    from dataclasses import dataclass, field
    import torch
    from torch import nn

    from chimera_ml.core.batch import Batch
    from chimera_ml.core.registry import INFERENCE_STEPS
    from chimera_ml.core.types import ModelOutput

    class ToyModel(nn.Module):
        def forward(self, batch):
            return ModelOutput(batch.inputs["x"])

    @dataclass
    class MyPredictionStep:
        value: float = 1.0
        on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = field(default=None, repr=False)

        def run(self, ctx):
            model = ToyModel().eval()
            batch = Batch({"x": torch.tensor([[self.value]])}, None)
            with torch.inference_mode():
                output = model(batch)
                if self.on_prediction is not None:
                    self.on_prediction(model, batch, output)
            ctx.set_artifact("predictions", output.preds.tolist())
            return ctx

    @INFERENCE_STEPS.register("my_prediction_step")
    def my_prediction_step(**params):
        return MyPredictionStep(**params)

Without XAI, configure the step directly:

    steps:
      - name: my_prediction_step
        params: {}

With XAI, wrap the same registered step:

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          modalities: [x]

Existing steps without XAI configuration continue to work unchanged.

Interventions are explicit. In 'mask' mode, the modality presence and temporal masks are set to false while unrelated masks remain unchanged; in 'remove' mode, the modality input and its masks are removed. Results use 'evaluated', 'already_missing', or 'unsupported' statuses, with null values when no valid comparison exists.

The policies are:

- 'zero': replace one input tensor with zeros;
- 'mask': disable its canonical presence and temporal masks;
- 'remove': remove its input key.

The caller chooses the policy compatible with the model. Masks are optional. Unexpected model errors propagate.

By default, predictions are compared directly. No softmax or task semantics are inferred. Pass 'score_fn(model_output)' for a task-specific score; it must return a batch-aligned tensor. The result contains baseline values, ablated values, delta = baseline - ablated, and absolute deltas. 'output_mask_key' can exclude invalid output positions. Diagnostics are copied only for explicitly requested 'diagnostic_keys'. 'validity_fn' runs after the ablated forward, so it can classify a result as unsupported but cannot prevent that forward.

These deltas measure local prediction sensitivity. They are not causal effects, SHAP values, calibrated confidence, or a composite trust score. Held-out quality comparison remains separate in 'evaluate_modality_quality'.
