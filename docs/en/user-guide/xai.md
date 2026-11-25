# Explainable AI (XAI)

CHIMERA-ML provides leave-one-modality-out analysis through `explain_batch` and the `xai_step` pipeline wrapper. For each sample and modality, the engine runs an intervention, selects a score, and compares it with the baseline. It is a local sensitivity diagnostic: it is not SHAP, a causal effect, calibrated confidence, accuracy, reliability, or a trust guarantee.

## Callback contract

`xai_step` wraps a registered prediction step. The wrapped step remains usable without XAI: its callback may be `None`. The exact optional field and invocation are:

```python
from collections.abc import Callable
from dataclasses import dataclass
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput

@dataclass
class PredictionStep:
    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = None

    def run(self, ctx):
        model = MyModel().to(ctx.device).eval()
        batch = make_batch(ctx)
        with torch.inference_mode():
            output = model(batch)
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        # Application-specific postprocessing belongs here.
        ctx.set_artifact("predictions", output.preds.detach().cpu().tolist())
        return ctx
```

The callback is called after the original `model(batch)` forward and before application-specific postprocessing. `XAIStep` does not require a callable to be installed in advance: it accepts `None`, temporarily installs its collector, invokes a previous callable if one exists, and restores the previous value in `finally`. A wrapped step must expose the attribute, otherwise the wrapper raises `TypeError`.

## Complete registered example

This deterministic example has two input modalities. The model sums them, so the expected baseline score is easy to check. The same registered step works directly and inside `xai_step`.

```python
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline

class ToyModel(nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        score = batch.inputs["audio"] + batch.inputs["video"]
        return ModelOutput(preds=score)

@dataclass
class ToyPredictionStep:
    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        model = ToyModel().to(ctx.device).eval()
        batch = Batch(
            inputs={
                "audio": torch.tensor([[2.0]], device=ctx.device),
                "video": torch.tensor([[1.0]], device=ctx.device),
            },
            targets=None,
            meta={"sample_ids": ["toy-0"]},
        )
        with torch.inference_mode():
            output = model(batch)                 # baseline score: [[3.0]]
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        ctx.set_artifact("predictions", output.preds.cpu().tolist())
        return ctx

@INFERENCE_STEPS.register("toy_prediction_step")
def toy_prediction_step(**params):
    return ToyPredictionStep(**params)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    config = InferenceConfig({
        "pipeline": {"name": "toy-xai"},
        "steps": [{
            "name": "xai_step",
            "params": {
                "step": {"name": "toy_prediction_step", "params": {}},
                "modalities": ["audio", "video"],
                "score_fn": lambda output: output.preds[:, 0],
            },
        }],
    })
    context = InferenceContext(root / "input.bin", root, "cpu", config.raw)
    result = build_inference_pipeline(config).run(context)
    artifact = result.get_artifact("xai")
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == 2.0
    assert artifact["samples"][0]["modalities"]["video"]["delta"] == 1.0
```

A YAML-loaded pipeline can use the same registered key, but arbitrary Python callables such as `score_fn` must be supplied by Python/plugin code rather than represented directly in YAML:

```yaml
steps:
  - name: xai_step
    params:
      step:
        name: toy_prediction_step
        params: {}
      modalities: [audio, video]
      ablation_policy: zero
```

Without XAI, use `name: toy_prediction_step` directly. The core registry contains `xai_step`; `toy_prediction_step` is illustrative and must be registered by the application/plugin.

## Interventions and raw explanations

The policies are explicit:

- `zero`: replace one modality tensor with zeros;
- `mask`: disable canonical presence/temporal masks and, when needed, add `<modality>_mask` filled with `False`;
- `remove`: remove the modality input and related masks, including nested legacy masks.

The caller chooses the policy compatible with the model. A missing modality receives `already_missing`; an intervention that cannot be applied receives `unsupported`. These entries have null `ablated`, `delta`, and `absolute_delta` values; they are not zero influence. Unexpected model errors propagate.

By default, the engine compares `output.preds` directly. It does not apply softmax or infer task semantics. `score_fn` must return a batch-aligned tensor. For every valid position:

```text
delta          = baseline_score - ablated_score
absolute_delta = abs(delta)
```

Thus a positive delta means that the selected score decreased after ablation; a negative delta means it increased. `output_mask_key` or a callable can restrict comparisons to valid output positions. Baseline and ablated valid positions are intersected. `validity_fn` runs after the ablated forward and may reject a sample, but cannot prevent that forward. `diagnostic_keys` copies selected batch-aligned values from `ModelOutput.aux` into each raw sample.

## Correct score-shape examples

Temporal meaning is never inferred automatically. The batch dimension is always first; choose `temporal_axis` only when the selected score has a time dimension.

### Scalar score per sample: `[B]`

For ordinary class/regression output `output.preds` shaped `[B, C]`, select one component:

```python
raw = explain_batch(
    model=model,
    batch=batch,
    score_fn=lambda output: output.preds[:, 0],  # [B]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw)  # no temporal_axis
```

The resulting score is `[B]`; it has no time axis. Supplying `temporal_axis=1` here is invalid.

### Temporal score: `[B, T]`

If the model directly returns one score per time position, leave the output as `[B, T]`:

```python
raw = explain_batch(model, batch, score_fn=lambda output: output.preds, modalities=["audio", "video"])
summary = summarize_explanations(raw, temporal_axis=1)
```

Here axis 0 is batch and axis 1 is time.

### Temporal score with output components: `[B, T, C]`

For output shaped `[B, T, C]`, select a component while preserving time:

```python
raw = explain_batch(
    model, batch,
    score_fn=lambda output: output.preds[:, :, 0],  # [B, T]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw, temporal_axis=1)
```

The selected score has `[B, T]`: axis 1 is time. If a score function instead returns `[B, T, C]`, `temporal_axis=1` still identifies the time dimension and `C` remains an output-component dimension. A `[B, C]` tensor is not temporal by default.

## Artifact schema and interpretation

`explain_batch` returns schema version 1. `xai_step` combines batches into an `xai` artifact with schema version 2, method `leave_one_modality_out`, the raw `samples`, and an analytical `summary`. A representative single-sample artifact is:

```json
{
  "schema_version": 2,
  "method": "leave_one_modality_out",
  "ablation_policy": "zero",
  "modalities": ["audio", "video"],
  "samples": [
    {
      "sample_id": "toy-0",
      "baseline": 3.0,
      "modalities": {
        "audio": {
          "status": "evaluated",
          "ablated": 1.0,
          "delta": 2.0,
          "absolute_delta": 2.0
        },
        "video": {
          "status": "evaluated",
          "ablated": 2.0,
          "delta": 1.0,
          "absolute_delta": 1.0
        }
      },
      "diagnostics": {},
      "metadata": {}
    }
  ],
  "batches": 1,
  "summary": {
    "status": "evaluated",
    "ranking": ["audio", "video"],
    "ties": [],
    "dominant_modality": "audio",
    "dominant_modalities": ["audio"],
    "total_samples": 1,
    "comparable_samples": 1,
    "sample_coverage": 1.0,
    "modalities": {
      "audio": {
        "mean_signed_delta": 2.0,
        "mean_absolute_delta": 2.0,
        "relative_influence": 0.6666666667,
        "rank": 1,
        "valid_comparison_count": 1,
        "common_comparison_count": 1,
        "comparable_sample_count": 1,
        "evaluated_sample_count": 1
      },
      "video": {
        "mean_signed_delta": 1.0,
        "mean_absolute_delta": 1.0,
        "relative_influence": 0.3333333333,
        "rank": 2,
        "valid_comparison_count": 1,
        "common_comparison_count": 1,
        "comparable_sample_count": 1,
        "evaluated_sample_count": 1
      }
    },
    "samples": [
      {
        "sample_id": "toy-0",
        "status": "evaluated",
        "ranking": ["audio", "video"],
        "ties": [],
        "dominant_modality": "audio",
        "common_comparison_count": 1,
        "sample_coverage": 1.0,
        "modalities": {
          "audio": {"mean_signed_delta": 2.0, "mean_absolute_delta": 2.0, "relative_influence": 0.6666666667, "rank": 1, "valid_comparison_count": 1, "common_comparison_count": 1},
          "video": {"mean_signed_delta": 1.0, "mean_absolute_delta": 1.0, "relative_influence": 0.3333333333, "rank": 2, "valid_comparison_count": 1, "common_comparison_count": 1}
        }
      }
    ]
  }
}
```

The exact artifact may contain additional fields such as `availability`, `temporal_axis`, or temporal summaries; the example shows the important contract and numerical relationship. `mean_signed_delta` is the mean of signed deltas; `mean_absolute_delta` is the mean of absolute deltas. Relative influence is:

```text
mean_absolute_delta(modality)
--------------------------------
sum(mean_absolute_delta(all comparable modalities))
```

The ranking orders modalities by mean absolute delta. `dominant_modality` is the sole largest modality. Equal largest values produce a `ties` group, set `dominant_modality` to `null`, and list all tied names in `dominant_modalities`. If all comparable magnitudes are zero, the status is `no_measurable_influence` and influence/rank fields are null.

A raw delta of `2.0` in the example means that removing audio lowered the selected score by 2.0; it does not mean audio contributed exactly two causal units. To understand changing influence, inspect per-sample `summary.samples` and, for temporal scores, each `temporal` position. The global summary uses only common valid positions across comparable modalities. `valid_comparison_count` is modality-specific; `common_comparison_count` counts positions retained for joint comparison. Missing/unsupported positions are excluded, not converted to zero.

Global aggregation gives samples equal weight: each sample first contributes its own mean signed and absolute delta, then those per-sample means are averaged. `sample_coverage` is the fraction of input samples with a common comparison. This prevents a sample with a long sequence from automatically outweighing a shorter sample.

## Optional diagnostics and temporal summaries

Set `diagnostic_keys` to copy selected, batch-aligned `ModelOutput.aux` values into raw samples. Diagnostics are observations from the baseline output; they are not automatically interpreted as importance.

Set `temporal_axis` on `xai_step` or `summarize_explanations` for scores with a list-like temporal dimension. `[B, T]` and `[B, T, C]` can be summarized by position. The result reports per-position comparisons, rankings, and dominant modalities. No input-frame, token, or segment attribution is inferred from this output sensitivity.

## Retrieving and exporting the XAI artifact

The standard JSON writers export only the input path and `predictions` artifact. They do not serialize every `InferenceContext` artifact, so `write_json_predictions_step` does not include `xai` automatically. In Python, retrieve and serialize it explicitly:

```python
import json
from pathlib import Path

result = build_inference_pipeline(config).run(context)
xai = result.get_artifact("xai")
if xai is None:
    raise ValueError("The pipeline did not produce an xai artifact")
output_path = Path("outputs/explanations.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(xai, indent=2, allow_nan=False), encoding="utf-8"
)
```

For a CLI pipeline, an application can register an output step without changing CHIMERA-ML:

```python
import json
from dataclasses import dataclass
from pathlib import Path
from chimera_ml.core.registry import INFERENCE_STEPS

@dataclass
class WriteXAIJsonStep:
    output_path: str = "xai.json"

    def run(self, ctx):
        artifact = ctx.get_artifact("xai")
        if artifact is None:
            raise ValueError("Expected xai artifact")
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        ctx.set_artifact("xai_output_path", path)
        return ctx

@INFERENCE_STEPS.register("write_xai_json_step")
def write_xai_json_step(**params):
    return WriteXAIJsonStep(**params)
```

Sequential configuration places the exporter after `xai_step`:

```yaml
steps:
  - name: xai_step
    params: {step: {name: toy_prediction_step, params: {}}}
  - name: write_xai_json_step
    params: {output_path: outputs/xai.json}
```

In a parallel pipeline, give nodes explicit IDs and make the exporter depend on XAI (and any prediction/preparation nodes it reads):

```yaml
pipeline: {name: xai-dag, parallel: true}
steps:
  - id: xai
    name: xai_step
    params: {step: {name: toy_prediction_step, params: {}}}
  - id: export
    name: write_xai_json_step
    after: [xai]
    params: {output_path: outputs/xai.json}
```

The exporter is documentation-only here: register and install it in the application/plugin before using the CLI. Do not add it to the framework registry source unless it belongs to the application.

## Cost and limitations

For each callback batch, XAI keeps one baseline forward and runs another forward for each selected modality. With `M` modalities, this is approximately `1 + M` forwards per batch, plus the surrounding prediction step. This can be expensive for large models or long sequences; select modalities and batches deliberately.

Results depend on the ablation policy, score function, masks, validity checks, model mode, and data distribution. They require no ground truth and measure sensitivity to the chosen intervention, not causal contribution or model quality. Heterogeneous scores with incompatible scales should not be aggregated. Temporal summaries describe output positions, not guaranteed input locations. Missing, unsupported, and invalid comparisons must remain visible when interpreting coverage.
