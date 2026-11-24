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


## From encoder/head fusion constructors

The built-in fusion constructors changed in the feature-only refactor. Earlier
versions accepted an encoders mapping and an externally supplied head; current
constructors accept input_dims, shared_dim, and output_dim instead.

Before:

~~~python
TOMARFusionModel(
    encoders=encoders,
    shared_dim=256,
    head=head,
    anchor_modality="text",
)
~~~

After:

~~~python
TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

Move precomputed features into Batch.inputs. The model now performs the input
projections and prediction layer internally. RAAV additionally keeps
frame_modalities and context_modality; TFS-DCMMOE may set gate_hidden; TFS may
set groups; TOMAR may set gate_hidden.

If each modality needs independent temporal padding and masks, use
multimodal_sequence_collate. Keep masking_collate for existing pipelines whose
legacy sequence and presence-mask behavior is sufficient. The built-in registry
keys remain tfs_fusion_model, tfs_dcmmmoe_fusion_model, raav_fusion_model, and
tomar_fusion_model; no other removed fusion registry name is documented without
history evidence.

No additional migration action is documented here without source/history
evidence.
