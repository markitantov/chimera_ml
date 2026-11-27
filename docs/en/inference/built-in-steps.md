# Built-in inference steps

The core package registers resolve_checkpoints_step,
write_json_predictions_step, print_json_predictions_step, and xai_step.

## xai_step

xai_step wraps a callback-capable prediction step and writes artifacts["xai"].
The nested step must call on_prediction(model, batch, output) after every
model forward. It may be passed as an object or as a registered nested mapping.

Parameters are step, modalities, score_fn, output_mask, and ablation_policy.
The policy is remove, mask, or zero. score_fn and output_mask callables are
Python-only values. See the XAI guide for the callback and artifact contract.
