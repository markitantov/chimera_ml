# Inference troubleshooting

- **Unknown inference step**: install the plugin in the CLI environment, run `chimera-ml plugins list`, then `chimera-ml registry list --type inference_steps`.
- **Input file not found**: `--input` must point to an existing file; check the current working directory and path spelling.
- **Invalid device**: `--device` and `runtime.device` accept only `auto`, `cpu`, or `cuda`. `auto` selects CUDA only when available.
- **No predictions artifact**: a prediction step must call `ctx.set_artifact("predictions", value)` before a print/write step.
- **Output is not created**: in sequential mode add `--output` or declare `write_json_predictions_step`; in parallel mode declare the writer and its `after` dependencies explicitly.
- **Checkpoint not found**: local paths must be files; HTTP(S) URLs need network access and a writable work/cache directory.
- **Checkpoint key missing**: make the resolver key and prediction step's `checkpoint_key` identical.
- **Dependency error**: check spelling and IDs in `after`; repeated step names need explicit IDs; remove cycles and self-dependencies.
- **Artifact conflict**: two unrelated DAG branches wrote the same key. Rename the artifact or add an ordered dependency and make the later step the owner.
- **XAI callback error**: the wrapped step must expose a callable-or-`None` `on_prediction` attribute and the model plus all nested modules must already be in evaluation mode.
- **Unsupported XAI comparison**: inspect missing modalities, masks, `output_mask_key`, and `validity_fn`; unsupported values are not zeros.
