# Troubleshooting

## Config section must be a mapping/list

Cause: ExperimentConfig.validate enforces the framework shape.

Fix: use mappings for data, model, train, loss, and optimizer; use lists of
{name, params} entries for metrics, callbacks, and logging. Run
chimera-ml validate-config again.

## Unknown key from a registry

Cause: the factory was not registered, usually because the plugin is not
installed or its entry point/import failed.

Fix: run chimera-ml plugins list, then registry list with the relevant --type.
Confirm the entry point and inspect warnings.

## Training reports a batch without targets

Cause: Trainer requires targets for optimization.

Fix: use a labeled training dataset. Use an inference pipeline for unlabeled
inference.

## Checkpoint or output errors in inference

Cause: local checkpoint refs must be files; remote refs require reachable
http/https URLs. In parallel mode --output cannot create an output step.

Fix: verify the path/network and declare write_json_predictions_step with its
after dependencies when pipeline.parallel is true.

## A callback cannot find its monitor

Cause: monitor names are exact keys in flat epoch logs.

Fix: use a key such as val/loss or the exact plugin metric key. For Optuna, the
same key must be available to sweep_target_callback.

## with_features fails

Cause: evaluation was asked to cache features but model output has no
aux["features"].

Fix: add that tensor in the model output or call Trainer.evaluate directly
with a feature_extractor.
