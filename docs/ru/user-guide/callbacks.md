# Callbacks

Callbacks получают lifecycle events Trainer: on_fit_start, on_epoch_start,
on_batch_end, on_epoch_end и on_fit_end.

Built-ins:

| Key | Purpose |
| --- | --- |
| checkpoint_callback | Save best monitored checkpoints и optional last.pt. |
| early_stopping_callback | Stop при отсутствии улучшения monitor. |
| snapshot_callback | Save config и/или source ZIP. |
| collect_predictions_callback | Export cached split predictions в CSV artifacts. |
| plot_confusion_matrix_callback | Export confusion matrices в PDF artifacts. |
| sweep_target_callback | Track best/last scalar values для Optuna. |
| telegram_notifier_callback | Completion notification через environment variables. |

Monitor — точный flat log key, например val/loss или plugin metric key.
Telegram callback по умолчанию читает TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID;
tokens не должны находиться в YAML.
