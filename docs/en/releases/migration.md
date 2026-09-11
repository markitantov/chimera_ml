# Migration guide

This page records actions for confirmed behavior changes. It is intentionally
not a duplicate changelog.

## From pre-0.2.1 configuration plumbing

The old yaml_config approach was removed in the 0.1.0 line. Use the current
ExperimentConfig/YAML sections and CLI builders; do not import or configure the
removed helper.

## From pre-0.2.1 plugin runtime

BuildContext was added in 0.2.1. A plugin that needs metadata from an earlier
component should declare context in its factory and use register/describe_context
rather than mutating the raw YAML or relying on ad hoc global state.

## Inference output in 0.2.4+

When pipeline.parallel is true, --output/-o no longer auto-creates
write_json_predictions_step. Declare the output step explicitly and define its
after dependencies.

## Sweep updates

Current sweep metadata is under log_path/experiment_name/_sweeps/sweep_id.
Grid trials use parameters or trials, while Optuna uses typed parameters and
adds sweep_target_callback. Review [Sweeps](../user-guide/sweeps.md) and the
canonical changelog when upgrading.

No additional migration action is documented here without source/history
evidence.
