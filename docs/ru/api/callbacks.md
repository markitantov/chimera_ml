# API callbacks

Callbacks расширяют жизненный цикл Trainer без изменения основного training loop.
Они получают события fit, epoch и batch и могут сохранять artifacts, запрашивать
остановку обучения, собирать predictions или отправлять уведомления.

Встроенные callbacks обычно создаются через реестр \`CALLBACKS\` в конфигурации
эксперимента. Напрямую объект удобно создавать в собственном loop или тестах.
Task-oriented примеры находятся в разделе [Callbacks](../user-guide/callbacks.md).

## Контракт жизненного цикла

\`BaseCallback\` задаёт порядок extension points: \`on_fit_start\`,
\`on_epoch_start\`, повторяющиеся вызовы \`on_batch_end\`, затем
\`on_epoch_end\` и \`on_fit_end\`. Hooks получают экземпляр \`Trainer\`, а также
epoch, global step и агрегированные logs, когда они предусмотрены контрактом.
Реализация должна явно управлять side effects и использовать состояние Trainer.

::: chimera_ml.callbacks.base.BaseCallback

## Checkpointing

\`CheckpointCallback\` наблюдает за одним scalar log, записывает \`last.pt\`
и хранит лучшие checkpoints в
\`<log_path>/<experiment_name>/<run_name>/checkpoints\`. Registry key:
\`checkpoint_callback\`.

::: chimera_ml.callbacks.checkpoint_callback.CheckpointCallback

### Factory

::: chimera_ml.callbacks.checkpoint_callback.checkpoint_callback

## Artifacts evaluation

\`CollectPredictionsCallback\` экспортирует cached predictions в CSV artifacts,
а \`PlotConfusionMatrixCallback\` строит confusion matrix для classification.
Оба callback используют MLflow-capable logger и включают caching predictions в
начале fit.

::: chimera_ml.callbacks.collect_predictions_callback.CollectPredictionsCallback

::: chimera_ml.callbacks.collect_predictions_callback.collect_predictions_callback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.PlotConfusionMatrixCallback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.plot_confusion_matrix_callback

## Управление обучением и observability

\`EarlyStoppingCallback\` устанавливает \`trainer.stop_training\` после заданного
patience. \`SweepTargetCallback\` сохраняет лучшее monitored value для sweep.
\`SnapshotCallback\` сохраняет snapshots source/config, а
\`TelegramNotifierCallback\` отправляет итоговое сообщение с credentials из
environment variables.

::: chimera_ml.callbacks.early_stopping_callback.EarlyStoppingCallback

::: chimera_ml.callbacks.early_stopping_callback.early_stopping_callback

::: chimera_ml.callbacks.sweep_target_callback.SweepTargetCallback

::: chimera_ml.callbacks.sweep_target_callback.sweep_target_callback

::: chimera_ml.callbacks.snapshot_callback.SnapshotCallback

::: chimera_ml.callbacks.snapshot_callback.snapshot_callback

::: chimera_ml.callbacks.telegram_notifier_callback.TelegramNotifierCallback

::: chimera_ml.callbacks.telegram_notifier_callback.telegram_notifier_callback
