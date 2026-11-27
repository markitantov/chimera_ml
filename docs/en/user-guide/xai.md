# Explainable AI step

The XAI step measures local prediction sensitivity to removing one modality.
For a selected score h, it computes

    delta_m = h(f(x_S)) - h(f(x_(S without m))).

The sign shows the direction of change. The absolute value shows sensitivity.
The step also preserves model diagnostics from ModelOutput.aux so attention,
frame weights, and residual coefficients can be compared with the measured
change.

## Callback contract

xai_step wraps a prediction step. The wrapped step must expose an
on_prediction attribute and call it immediately after each model forward:

~~~python
@dataclass
class FusionStep:
    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = None

    def run(self, ctx):
        model = load_model(ctx).eval()
        batch = make_batch(ctx)
        with torch.inference_mode():
            output = model(batch)
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        write_predictions(ctx, output)
        return ctx
~~~

The callback receives the same model, Batch, and ModelOutput used for the
ordinary prediction. xai_step temporarily installs its collector, invokes the
previous callback if one existed, and restores it even when the wrapped step
fails. The callback is therefore executed per batch, before application-level
postprocessing.

## Configuration

The nested step can be an already constructed object or a registered step
mapping:

~~~yaml
- name: xai_step
  params:
    step:
      name: fusion_step
      params: {}
    modalities: [audio, video]
    ablation_policy: remove
~~~

The factory builds the nested registered step. Python callables such as
score_fn and output_mask must be supplied from Python code; they cannot be
written as YAML values.

## Parameters

- step: callback-capable prediction step.
- modalities: selected input names; omitted means every modality in the batch.
- score_fn: optional callable ModelOutput -> tensor. Without it, output.preds is
  compared.
- output_mask: optional mask name or callable returning a tensor with the score
  shape or its shape without the last component.
- ablation_policy: remove, mask, or zero.

The model must be in evaluation mode, including all child modules. The default
remove policy removes the selected input and its canonical masks. mask clears
its canonical masks. zero keeps the input and replaces its values with zeros.

Models may implement validate_intervention(original, intervened, modality,
policy) -> str | None. A returned message marks the comparison unsupported.
The step also protects a model anchor from remove and mask interventions.

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

Statuses are evaluated, already_missing, or unsupported. Invalid output
positions are serialized as null. The summary helper returns mean signed delta,
mean absolute delta, and a ranking by absolute delta:

~~~python
summary = summarize_explanations(result.get_artifact("xai"))
~~~

Quality on a labelled control set is intentionally a separate metric
operation. The XAI artifact contains the baseline and ablated predictions
needed to calculate that comparison without imposing a task-specific metric.

## Model diagnostics

TFS exposes pair and group attention and weights. RAAV exposes frame_weights,
frame_mask, context attention, and context_delta. TOMAR exposes rho_<name> and
delta_<name>. These values describe the internal fusion process; XAI does not
relabel them as causal importance.
