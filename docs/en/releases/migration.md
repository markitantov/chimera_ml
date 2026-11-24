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


## Current feature-only fusion architectures

The current TFS, TFS-DCMMOE, RAAV, and TOMAR classes are new feature-only
architectures. They accept `input_dims`, `shared_dim`, and `output_dim`, read
precomputed features from `Batch.inputs`, and create their projections and
prediction layers internally. They are not a backward-compatible continuation
of an earlier `TOMARFusionModel(encoders=..., head=...)` public API; that
example is not a valid migration from a released `main` implementation.

Use the model-specific guides for the current constructor fields. RAAV also
requires `frame_modalities` and `context_modality`; TFS-DCMMOE may set
`gate_hidden`, TFS may set `groups`, and TOMAR may set `gate_hidden`.

## Intentional removal of legacy fusion models

The refactor intentionally removed the legacy classes
`FeatureFusionModel`, `PredictionFusionModel`, `GatedFusionModel`, and
`GatedPredictionFusionModel`, together with their registry keys
`feature_fusion_model`, `prediction_fusion_model`, `gated_fusion_model`, and
`gated_prediction_fusion_model`. There is no direct drop-in replacement for
these classes. Choose one of the four current feature-only fusion models when
its architecture matches the task, and migrate the input and output contract
explicitly; old constructor parameters and submodel composition are not
compatible.

If each modality needs independent temporal padding and masks, use
`multimodal_sequence_collate`. During CLI `train` and `eval`, a present
top-level `collate:` section is resolved with `build_collate()` and injected
into the datamodule factory as `collate_fn`. A custom datamodule must declare a
keyword-capable `collate_fn` parameter to support this integration. If the
section is absent, the CLI does not build or inject a collate and preserves the
datamodule's existing default, including legacy `MaskingCollate` behavior.

There is no direct drop-in replacement for a removed legacy registry key.
