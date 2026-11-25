# Callbacks в training workflow

Callbacks получают lifecycle events Trainer: `on_fit_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, `on_fit_end`. Монитор должен быть точным flat log key.

## Checkpoint, early stopping и snapshot

```yaml
callbacks:
  - name: checkpoint_callback
    params:
      log_path: logs
      monitor: val/mae
      mode: min
      save_top_k: 1
      save_last: true
      filename_template: "epoch={epoch}_{monitor}={value:.4f}.pt"
  - name: early_stopping_callback
    params:
      monitor: val/mae
      mode: min
      patience: 5
      min_delta: 0.0
  - name: snapshot_callback
    params:
      log_path: logs
      include: [src]
      save_code_zip: true
      save_config: true
```

Пример предполагает metric `mae` на split `val`. Checkpoint создаёт `logs/<experiment>/<run>/checkpoints`; snapshot создаёт `code.zip` и копию config. CLI inject-ит experiment/run names и config path.

## Predictions и confusion matrix

Оба callback требуют MLflow logger и включают evaluation cache:

```yaml
logging:
  - name: mlflow_logger
    params: {tracking_uri: sqlite:///logs/mlflow.db}
callbacks:
  - name: collect_predictions_callback
    params:
      splits: [val, test]
      artifact_path: predictions
      filename_template: preds_epoch_{epoch}.csv
      task: classification
      include_probs: true
  - name: plot_confusion_matrix_callback
    params:
      splits: [val]
      class_names: [negative, neutral, positive]
      artifact_path: figures
      filename_template: confusion_matrix_epoch_{epoch}.pdf
```

Prediction callback пишет CSV в `predictions/<split>`: для regression — `pred_*`/`target_*`, для classification — `pred_class`, probabilities и target. Confusion callback требует classification-compatible outputs и пишет PDF в `figures/<split>`. Нужен `matplotlib`.

## Telegram и sweep target

```bash
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
```

```yaml
- name: telegram_notifier_callback
  params:
    token_env_field: TELEGRAM_BOT_TOKEN
    chat_id_env_field: TELEGRAM_CHAT_ID
    monitor: val/mae
    mode: min
    include_last_logs: true
    include_best_logs: true
    request_timeout_sec: 15
```

Notifier отправляет сообщение на `on_fit_end`, требует network и requests. Tokens не записывайте в YAML. `sweep_target_callback` обычно добавляется автоматически для Optuna и отслеживает best scalar, но не останавливает training и не сохраняет files.

Dependency checklist: monitored key должен существовать; prediction/figure callbacks требуют cached labeled outputs и MLflow; snapshot copy config требует CLI config path или explicit `config_path` при direct API use.
