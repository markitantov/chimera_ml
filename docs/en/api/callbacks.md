# Callbacks API

Callbacks extend the trainer lifecycle without changing the training loop.
They observe fit, epoch, and batch events and can persist artifacts, request
early stopping, collect predictions, or send notifications.

Built-in callbacks are normally created from the \`CALLBACKS\` registry in the
experiment configuration. Instantiate one directly when writing a custom loop
or testing an implementation. For task-oriented configuration examples, see
[Callbacks](../user-guide/callbacks.md).

## Lifecycle contract

\`BaseCallback\` defines the extension points and their call order:
\`on_fit_start\`, \`on_epoch_start\`, repeated \`on_batch_end\` calls,
\`on_epoch_end\`, and \`on_fit_end\`. Hooks receive the \`Trainer\` instance and,
where applicable, the current epoch, global step, and aggregated log values.
Implementations should keep hook side effects explicit and use the trainer
state supplied by the callback contract.

::: chimera_ml.callbacks.base.BaseCallback

## Checkpointing

\`CheckpointCallback\` monitors one scalar log entry, writes \`last.pt\` when
enabled, and retains the best checkpoints under
\`<log_path>/<experiment_name>/<run_name>/checkpoints\`. Its registry key is
\`checkpoint_callback\`.

::: chimera_ml.callbacks.checkpoint_callback.CheckpointCallback

### Factory

::: chimera_ml.callbacks.checkpoint_callback.checkpoint_callback

## Evaluation artifacts

\`CollectPredictionsCallback\` exports cached predictions as CSV artifacts, while
\`PlotConfusionMatrixCallback\` renders classification confusion matrices. Both
use the trainer's MLflow-capable logger and request prediction caching at fit
start.

::: chimera_ml.callbacks.collect_predictions_callback.CollectPredictionsCallback

::: chimera_ml.callbacks.collect_predictions_callback.collect_predictions_callback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.PlotConfusionMatrixCallback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.plot_confusion_matrix_callback

## Training control and observability

\`EarlyStoppingCallback\` sets \`trainer.stop_training\` after the configured
patience. \`SweepTargetCallback\` records the best monitored value for sweep
reporting. \`SnapshotCallback\` saves source/config snapshots, and
\`TelegramNotifierCallback\` sends a final message using environment-provided
credentials.

::: chimera_ml.callbacks.early_stopping_callback.EarlyStoppingCallback

::: chimera_ml.callbacks.early_stopping_callback.early_stopping_callback

::: chimera_ml.callbacks.sweep_target_callback.SweepTargetCallback

::: chimera_ml.callbacks.sweep_target_callback.sweep_target_callback

::: chimera_ml.callbacks.snapshot_callback.SnapshotCallback

::: chimera_ml.callbacks.snapshot_callback.snapshot_callback

::: chimera_ml.callbacks.telegram_notifier_callback.TelegramNotifierCallback

::: chimera_ml.callbacks.telegram_notifier_callback.telegram_notifier_callback
