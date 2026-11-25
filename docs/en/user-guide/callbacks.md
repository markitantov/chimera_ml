# Training callbacks in practice

For the complete callback catalog and lifecycle details, see [Callbacks](../components/callbacks/index.md). This page focuses on workflow setup.

Callbacks observe the Trainer lifecycle: `on_fit_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, and `on_fit_end`. Configure them as a list. A callback that monitors a metric needs the exact flat key emitted at epoch end.

## A practical callback set

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

The example assumes a metric named `mae` on a validation split named `val`. If a plugin callback produces `val_framewise/va_ccc_metric`, use that exact key instead. `checkpoint_callback` creates `logs/<experiment>/<run>/checkpoints`; `snapshot_callback` creates `code.zip` and copies the config into the run directory. The CLI injects experiment/run names and the snapshot config path when appropriate.

## Prediction collection and confusion matrices

Both callbacks require an MLflow logger because they upload artifacts, and both enable evaluation cache on fit start:

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
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

`collect_predictions_callback` writes per-sample CSV artifacts under `predictions/<split>` and can read IDs from `Batch.meta["sample_meta"]`. For regression it writes `pred_*` and `target_*`; for classification it writes `pred_class`, optional probabilities, and a scalar target. `plot_confusion_matrix_callback` needs classification-like predictions/targets and writes PDFs under `figures/<split>`. Neither callback creates artifacts if no MLflow logger is configured.

## Telegram notifications

The notifier sends one message at `on_fit_end` and requires both environment variables by default:

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

It also requires network access and the `requests` dependency. Use custom environment variable names with the two `*_env_field` parameters. Never put tokens in YAML.

## Sweep objective callback

`sweep_target_callback` is normally inserted automatically for Optuna sweeps. It tracks the best scalar monitor and does not itself stop training or save files. Add it manually only when a caller needs the callback's tracked result.

## Dependency checklist

- Checkpoint, early stopping, scheduler monitor, and Telegram monitor must be present in epoch logs.
- Prediction and confusion callbacks require cached labeled outputs; they set `collect_cache: true`, but an MLflow logger is still required for artifact upload.
- Confusion matrices require classification-compatible prediction/target shapes and `matplotlib`.
- Snapshot config copying requires the CLI-provided config path; direct API use must set `config_path` itself.
