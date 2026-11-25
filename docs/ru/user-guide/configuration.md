# Конфигурация эксперимента

Training/evaluation YAML содержит framework sections и plugin-specific params. Core проверяет форму sections, а factory отвечает за собственные params. Для `train`/`eval` обязательны `data`, `model`, `train`, `loss` и `optimizer`; для `train`/`sweep` нужен `experiment_info.params.experiment_name`.

## Полный plugin-backed пример

Ниже — полный `examples/va_estimation/configs/multimodal_train.yaml` из репозитория. После установки VA plugin и подготовки paths он внутренне согласован. Keys `fusion_va_datamodule`, `fusion_model`, `fusion_ccc_mse_loss`, `va_ccc_metric` и `fusion_framewise_callback` регистрируются plugin, не core.

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
      face: {path: "data/features/face_features", emb_size: 256}
      qwen_mm: {path: "data/features/qwen_mm", emb_size: 256}
      qwen_v: {path: "data/features/qwen_v", emb_size: 128}

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
    params: {monitor: "val_framewise/va_ccc_metric", mode: "max", patience: 3}
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

Для этого файла нужны editable plugin install, feature/annotation paths из `data.params` и Telegram credentials, если callback оставлен. Иначе удалите notifier. Проверка:

```bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

## Секции

| Section | Required | Значение |
| --- | --- | --- |
| `seed` | нет, default `0` | Передаётся seed helper. |
| `experiment_info.params.experiment_name` | для `train`/`sweep` | Identity experiment и output paths. |
| `data` | да для train/eval | `name` и `params` registered DataModule factory. |
| `collate` | нет | Top-level registered collate, injected как `collate_fn`. |
| `model` | да | Registered model и constructor params. |
| `train.params` | да | `TrainConfig`: epochs, device, precision, clipping, loader mode, scheduler и cache. |
| `loss` | да | Registered loss factory. |
| `metrics` | нет | Список registered metrics. |
| `optimizer` | да | Registered optimizer. |
| `scheduler` | нет | Registered scheduler и его PyTorch params. |
| `callbacks` | нет | Lifecycle callbacks. |
| `logging` | нет | `console_file_logger`, `mlflow_logger` или plugin loggers. |

Named item имеет вид:

```yaml
name: registry_key
params:
  option: value
```

Core не проверяет plugin paths, feature dimensions и labels. Не называйте config self-contained, если нужны external data.

`device` обычно `cuda`/`cpu`; `mixed_precision` действует только на CUDA. `log_path` задаётся logger/callback params, а не `train.params`. Logger создаёт `<log_path>/<experiment>/<run>/<log_file>`, checkpoint/snapshot используют ту же experiment/run directory.
