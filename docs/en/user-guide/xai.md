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

These deltas measure local prediction sensitivity. They are not causal effects, SHAP values, calibrated confidence, or a composite trust score.


## Influence summary

After collecting raw explanations, 'XAIStep' adds a deterministic 'summary' to the 'xai' artifact. Summary generation reads only the raw samples and performs no model forwards.

For each comparable modality:

- 'mean_signed_delta = mean(delta)' describes the direction of the selected score change;
- 'mean_absolute_delta = mean(abs(delta))' describes measured sensitivity;
- 'relative_influence = mean_absolute_delta / sum(mean_absolute_delta)' across the common comparable modalities;
- 'rank' orders modalities by descending mean absolute delta.

Only common valid output positions are used for a joint ranking. Nulls, 'already_missing', 'unsupported', and insufficient comparisons are excluded, never treated as zero. Samples receive equal weight in the global aggregation: each sample contributes its own mean before global means are calculated. Relative influence is normalized only after those comparable absolute means are aggregated.

Ties are listed in 'ties'; 'dominant_modality' is null for ties and for all-zero influence. A zero-valued evaluated delta remains distinguishable from unavailable or unsupported data.

A shortened artifact shape is:

    {
      "schema_version": 2,
      "method": "leave_one_modality_out",
      "ablation_policy": "zero",
      "modalities": ["audio", "video"],
      "summary": {
        "status": "evaluated",
        "ranking": ["audio", "video"],
        "dominant_modality": "audio",
        "dominant_modalities": ["audio"],
        "total_samples": 2,
        "comparable_samples": 2,
        "sample_coverage": 1.0,
        "modalities": {
          "audio": {
            "status": "evaluated",
            "mean_signed_delta": 0.3,
            "mean_absolute_delta": 0.3,
            "relative_influence": 0.75,
            "rank": 1,
            "valid_comparison_count": 2,
            "common_comparison_count": 2
          },
          "video": {
            "status": "evaluated",
            "mean_signed_delta": -0.1,
            "mean_absolute_delta": 0.1,
            "relative_influence": 0.25,
            "rank": 2,
            "valid_comparison_count": 2,
            "common_comparison_count": 2
          }
        },
        "samples": []
      },
      "samples": []
    }

The raw 'samples' remain available beside 'summary'. A positive signed delta means the selected score decreased after ablation when 'delta = baseline - ablated'; a negative value means it increased. For example, an audio delta of '0.3' indicates a decrease of '0.3', while a text delta of '-0.2' indicates an increase of '0.2'. Missing or unsupported modality entries retain their status and have null influence fields.

To localize explicitly temporal outputs, set 'temporal_axis' on 'xai_step', relative to the batch-aligned score tensor:

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          modalities: [audio, video]
          temporal_axis: 1

'[B, T]' and '[B, T, C]' can therefore be analyzed by output position. '[B, C]' is analyzed temporally only if the caller explicitly sets 'temporal_axis: 1'; no tensor dimension is treated as time by default. This is output sensitivity, not frame-, token-, or segment-level input attribution. Window identifiers and timing metadata are preserved only when the application supplies them.

The selected 'score_fn(model_output)' must produce comparable values. Aggregating heterogeneous output components with different scales is not meaningful; select or transform a meaningful score first. Influence is relative to the selected intervention and score. It is not a probability, causal contribution, confidence, accuracy, reliability, or trust score. No ground truth is required, and summary generation adds no forwards.
