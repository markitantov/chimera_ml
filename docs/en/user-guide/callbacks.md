# Callbacks

Callbacks receive lifecycle events from Trainer: on_fit_start, on_epoch_start,
on_batch_end, on_epoch_end, and on_fit_end.

Built-ins:

| Key | Purpose |
| --- | --- |
| checkpoint_callback | Save best monitored checkpoints and optionally last.pt. |
| early_stopping_callback | Stop after monitored values fail to improve. |
| snapshot_callback | Save config and/or a source ZIP at fit start. |
| collect_predictions_callback | Export cached split predictions as CSV artifacts. |
| plot_confusion_matrix_callback | Export cached confusion matrices as PDF artifacts. |
| sweep_target_callback | Track best/last scalar values for Optuna. |
| telegram_notifier_callback | Send completion notification using environment variables. |

Callbacks monitor exact flat log keys such as val/loss or plugin-produced keys.
The key must exist for checkpointing, early stopping, or sweep objectives to
act.

The Telegram callback reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID by default.
Do not place tokens in YAML or documentation.
