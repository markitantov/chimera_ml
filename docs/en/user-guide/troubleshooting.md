# Training troubleshooting

## Configuration and registries

- **“Section must be a mapping/list”**: use the expected top-level shape. `data`, `model`, `loss`, and `optimizer` are mappings; `metrics`, `callbacks`, and `logging` are lists of named items.
- **Missing experiment name**: add `experiment_info.params.experiment_name` for `train` and `sweep`.
- **Unknown registry key**: install the plugin, run `chimera-ml plugins list`, then inspect the relevant `registry list --type ...`. Registration happens at process startup.
- **Top-level `collate` conflict**: do not also set `data.params.collate_fn`; the builder rejects both forms.

## Data and shapes

- **Dataset paths fail**: `validate-config` does not check filesystem paths. Confirm every plugin path under `data.params` and that the process working directory is the one expected by the YAML.
- **No targets**: training requires `Batch.targets`; fix the dataset or collate. Unlabeled data belongs in evaluation/inference.
- **Shape mismatch**: inspect modality feature dimensions, target dimensions, and the model's input contract. `MultimodalSequenceCollate` requires rank-one `[D]` or rank-two `[T, D]` modality samples with a consistent feature dimension per modality.
- **Mask mismatch**: temporal masks must match `[B, T]`; explicit sample masks must match the unpadded sequence length. Missing inputs are zero-filled but should remain marked unavailable by presence masks.

## Devices and numerical errors

- **CUDA requested but unavailable**: `doctor` reports `cuda_available`; the Trainer resolves to CPU. Install a compatible CUDA PyTorch wheel/driver if GPU execution is required.
- **NaN or Inf**: the Trainer raises a finite-value error. Check input normalization, targets, loss scale, learning rate, and masks before increasing complexity.
- **Mixed precision behaves like CPU**: AMP is only enabled when the resolved device is CUDA.

## Monitors, callbacks, and outputs

- **Checkpoint/early stopping says monitor is missing**: use the exact epoch key, such as `val/loss`, `val/mae`, or a plugin key like `val_framewise/va_ccc_metric`.
- **Prediction/figure artifacts are absent**: those callbacks require an MLflow logger and cached labeled outputs; inspect the callback split selector and `collect_cache`.
- **Telegram callback fails at fit start**: export the configured token and chat ID environment variables. No credentials are read from YAML unless the variable names point to them.

## Checkpoint compatibility

Evaluation loads `model_state_dict` strictly. Use the same model registry key, constructor parameters, modality names, and feature dimensions that produced the checkpoint. A raw state dictionary is accepted, but optimizer/scheduler metadata is not available from it.
