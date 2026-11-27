# Explainable AI step

The XAI step measures local prediction sensitivity to excluding one modality.
For a selected score h, it computes

    delta_m = h(f(x_S)) - h(f(x_(S without m))).

The delta is signed: its direction shows how the selected score changes when a
modality is excluded. The artifact keeps individual deltas, frame-level
results, output masks, and model diagnostics. It does not calculate rankings,
aggregate summaries, or dataset quality metrics.

## Callback contract

xai_step wraps a prediction step. The wrapped step must expose an
on_prediction attribute and call it immediately after each model forward.
The callback receives the same model, Batch, and ModelOutput used by the
ordinary prediction. xai_step temporarily installs its collector, invokes the
previous callback if one existed, and restores it even when the wrapped step
fails.

The model must be in evaluation mode. explain_batch runs under
torch.inference_mode(). A model may optionally implement
validate_intervention(original, intervened, modality, policy) -> str | None.
A returned reason marks that comparison unsupported and skips the intervened
forward. Unexpected validation or model errors are propagated.

## Configuration

~~~yaml
- name: xai_step
  params:
    step:
      name: fusion_step
      params: {}
    modalities: [audio, video]
    ablation_policy: remove
~~~

The nested step can also be an already constructed object. score_fn and
output_mask are Python callables and cannot be written as YAML values.

## Parameters

- step: callback-capable prediction step.
- modalities: selected input names; omitted means every modality found in the
  processed batches.
- score_fn: optional callable ModelOutput -> tensor. Without it, output.preds is
  compared.
- output_mask: optional mask name or callable returning a tensor with the score
  shape or its shape without the last component.
- ablation_policy: remove or mask.

remove deletes the selected input and its relevant masks. mask preserves the
input and clears existing relevant masks; if none exists, it creates a false
canonical modality mask. Both policies perform actual exclusion. There is no
zero-value intervention.

## Artifact

The context receives artifacts["xai"]:

~~~python
{
    "ablation_policy": "remove",
    "modalities": ["audio", "video"],
    "samples": [
        {
            "sample_id": "sample-0",
            "baseline": 0.42,
            "modalities": {
                "audio": {
                    "status": "evaluated",
                    "reason": None,
                    "delta": 0.08,
                }
            },
            "aux": {"frame_weights": [[0.6, 0.4]]},
        }
    ],
}
~~~

Statuses are evaluated, already_missing, or unsupported. Invalid output positions are represented by null values; output masks are not serialized. For several processed batches, the modality
list is the deterministic union of the configured names and names observed in
the callback results.

## Model diagnostics

TFS exposes directed pair and group attention and weights. RAAV exposes frame
weights, frame masks, context attention, compression attention, and context
corrections. TOMAR exposes residual weights and corrections. These values
describe the internal fusion process; XAI does not relabel them as causal
importance.
