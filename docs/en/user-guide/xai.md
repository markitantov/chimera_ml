# Explainable AI (XAI)

CHIMERA-ML provides leave-one-modality-out analysis through `explain_batch` and the `xai_step` pipeline wrapper. For each sample and modality it runs an intervention, selects a score, and compares the result with the baseline. This is local prediction-sensitivity analysis—not SHAP, a causal effect, confidence, accuracy, reliability, or trust guarantee.

## Requirements and callback contract

The model and all nested modules must already be in `eval()` mode. `explain_batch` uses `torch.inference_mode()`. A wrapped prediction step must expose a callable `on_prediction` callback and invoke it after every model forward. `xai_step` temporarily installs its callback, invokes any previous callback, and restores it in `finally`.

```python
from dataclasses import dataclass, field
from collections.abc import Callable
import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput

@dataclass
class PredictionStep:
    on_prediction: Callable | None = field(default=None, repr=False)

    def run(self, ctx):
        model = MyModel().to(ctx.device).eval()
        batch = make_batch(ctx)
        with torch.inference_mode():
            output = model(batch)
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        ctx.set_artifact("predictions", output.preds.detach().cpu().tolist())
        return ctx
```

The callback-compatible step remains an ordinary prediction step when XAI is not configured.

## Direct API

```python
from chimera_ml.inference import explain_batch, summarize_explanations

raw = explain_batch(
    model=model,
    batch=batch,
    modalities=["audio", "video"],
    ablation_policy="mask",
    score_fn=lambda output: output.preds[:, 0],
    output_mask_key="valid_mask",
)
summary = summarize_explanations(raw, temporal_axis=1)
```

`modalities` defaults to the keys in `batch.inputs`. The policies are:

- `zero`: replace the selected tensor with zeros;
- `mask`: disable canonical presence/temporal masks and, when needed, add `<modality>_mask` filled with `False`;
- `remove`: remove the input and related masks, including nested legacy masks.

Missing inputs receive `already_missing`; unsupported interventions receive `unsupported`. Their values remain null and are not treated as zero influence.

The default score is `output.preds` directly; no softmax or task semantics are inferred. `score_fn` must return a batch-aligned tensor. `delta = baseline - ablated`, so a positive signed delta means the selected score decreased after ablation. `output_mask_key` restricts comparisons to valid output positions. `validity_fn` runs after the ablated forward and may reject a result, but cannot prevent that forward. `diagnostic_keys` selects batch-aligned values from `ModelOutput.aux`.

## Pipeline configuration

The nested step is built through `INFERENCE_STEPS` and must support the callback:

```yaml
- name: xai_step
  params:
    step:
      name: my_prediction_step
      params: {}
    modalities: [audio, video]
    ablation_policy: zero
    diagnostic_keys: [features]
    temporal_axis: 1
```

`xai_step` publishes an `xai` artifact with schema version 2, method `leave_one_modality_out`, the raw batch results, and a summary. The raw result from `explain_batch` has schema version 1. `score_fn` and `validity_fn` are Python callables; arbitrary callables cannot be represented by YAML, so provide them from Python or a plugin.

## Summary and temporal outputs

Summary generation performs no model forwards. It calculates mean signed and absolute deltas, relative influence, ranks, ties, and dominant modality using only common valid positions. Missing, unsupported, null, and insufficient comparisons are excluded rather than converted to zero. Possible diagnostic statuses include `insufficient_data`, `no_common_comparison`, `unavailable`, `no_measurable_influence`, and `insufficient_common_support`.

`temporal_axis` is a positive batch-relative axis for output shaped `[B, T]` or `[B, T, C]`. A `[B, C]` output is not treated as temporal unless the caller explicitly supplies an axis. This localizes output sensitivity; it does not claim frame-, token-, or segment-level input attribution. No ground truth is required.

The complete public API is in the [Inference API reference](../api/inference.md).
