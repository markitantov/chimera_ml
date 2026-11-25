# Configuring an experiment

A training/evaluation YAML has framework sections and plugin-specific parameters. The core validates the section shape; each selected factory validates or uses its own `params` signature. Required sections for `train` and `eval` are `data`, `model`, `train`, `loss`, and `optimizer`; `experiment_info.params.experiment_name` is required by `train` and `sweep`.

## Complete plugin-backed example

The following is an adapted plugin-backed example, not an exact copy of the shipped VA YAML. It is internally consistent after installing the VA plugin and replacing the external paths described below.

```yaml
seed: 0
experiment_info:
  params:
    experiment_name: "va_estimation_fusion"
    run_name: "fusion"
    datetime_format: "%Y-%m-%d_%H-%M"

data:
  name: "fusion_va_datamodule"
  params:
    audio:
      path: "data/features/audio_features_w_fcl_new"
      emb_size: 256
    use_predictions: true
    target_source_modality: "face"
    target_source_field: "label"
    frame_index_offset: 1
    mask_audio: true
    audio_min_open_sec: 1
    audio_min_coverage_ratio: 0.8
    pin_memory: false
    num_workers: 1
    modalities:
      face:
        path: "data/features/face_features"
        emb_size: 256
      qwen_mm:
        path: "data/features/qwen_mm"
        emb_size: 256
      qwen_v:
        path: "data/features/qwen_v"
        emb_size: 128

model:
  name: "fusion_model"
  params:
    use_predictions: true
    context_modality: "audio"
    frame_modalities: ["face", "qwen_mm"]
    hidden: 256
    dropout: 0.1
    num_heads: 4
    num_temporal_layers: 1
    num_context_latents: 4
    modality_priors: {face: 0.15, qwen_mm: 0.85}

train:
  params:
    epochs: 10
    mixed_precision: true
    device: "cuda"
    log_every_steps: 50
    grad_clip_norm: 1.0
    use_scheduler: true
    scheduler_step_per_epoch: true
    scheduler_monitor: "val_framewise/va_ccc_metric"
    collect_cache: false

optimizer:
  name: "adamw_optimizer"
  params: {lr: 5.0e-5, weight_decay: 0.01}

loss:
  name: "fusion_ccc_mse_loss"
  params: {ccc_weight: 1, mse_weight: 0.01, v_weight: 0.7, a_weight: 0.3}

metrics:
  - name: "va_ccc_metric"
    params: {}

logging:
  - name: "console_file_logger"
    params:
      log_path: "logs"
      log_file: "train.log"
      console_level: "INFO"
      file_level: "INFO"
  - name: "mlflow_logger"
    params:
      tracking_uri: "sqlite:///logs/mlflow.db"

callbacks:
  - name: "fusion_framewise_callback"
    params:
      loader_name: "val"
      log_prefix: "val_framewise"
      ann_root: "data/labels/VA_Estimation_Challenge"
      ann_split_dir: "Validation_Set"
      overlap_strategy: "mean"
      dump_pickle: false
      frame_index_offset: 1
  - name: "early_stopping_callback"
    params:
      monitor: "val_framewise/va_ccc_metric"
      mode: "max"
      patience: 3
  - name: "checkpoint_callback"
    params:
      log_path: "logs"
      filename_template: "epoch={epoch}_{monitor}={value:.4f}.pt"
      monitor: "val_framewise/va_ccc_metric"
      mode: "max"
      save_top_k: 1
      save_last: true
  - name: "snapshot_callback"
    params:
      log_path: "logs"
      include: ["src"]
      save_code_zip: true
      save_config: true
  - name: "telegram_notifier_callback"
    params:
      token_env_field: "TELEGRAM_BOT_TOKEN"
      chat_id_env_field: "TELEGRAM_CHAT_ID"
      message_prefix: "✅ Training finished"
      monitor: "val_framewise/va_ccc_metric"
      mode: "max"
      request_timeout_sec: 15
```

Prerequisites for this exact file are the editable plugin install, feature paths configured under `data.params`, annotation paths configured as `ann_root` and `ann_split_dir` inside `fusion_framewise_callback.params`, and any Telegram credentials if the notification callback is retained. Remove that callback if notifications are not required. Validate it with:

```bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

## Section contract

| Section | Required | Meaning |
| --- | --- | --- |
| `seed` | no, default `0` | Passed to the framework seed helper. |
| `experiment_info.params.experiment_name` | for `train`/`sweep` | Experiment identity used in run names and output paths. |
| `data` | yes for train/eval | `name` and `params` for a registered DataModule factory. |
| `collate` | no | Top-level registered collate injected as `collate_fn` when the DataModule factory declares it. |
| `model` | yes | Registered model factory and its constructor parameters. |
| `train.params` | yes | `TrainConfig` fields such as `epochs`, `device`, precision, clipping, loader mode, scheduler behavior, and cache. |
| `loss` | yes | Registered loss factory. |
| `metrics` | no | List of registered metric items. |
| `optimizer` | yes | Registered optimizer; core direct builders default to `adamw_optimizer` if no explicit config is used, but an explicit entry is clearer. |
| `scheduler` | no | Registered scheduler and its PyTorch scheduler parameters. |
| `callbacks` | no | Lifecycle callback list. |
| `logging` | no | `console_file_logger`, `mlflow_logger`, or plugin loggers. |

Every named item has this shape:

```yaml
name: registry_key
params:
  option: value
```

The framework does not check whether plugin paths, feature dimensions, labels, or plugin-specific parameters are semantically correct. Use the plugin implementation and a small validation run to verify those details; do not call a config self-contained when external data is required.

## Device and output settings

`train.params.device` is normally `cuda` or `cpu`; the Trainer falls back to CPU when CUDA is unavailable. `mixed_precision` only has an effect on CUDA. `log_path` belongs to logger/callback parameters, not to `train.params`. The console logger creates `<log_path>/<experiment_name>/<run_name>/<log_file>`, while checkpoint and snapshot callbacks create their artifacts beneath the same experiment/run directory.
