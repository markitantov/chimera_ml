# API колбэков

Колбэки расширяют жизненный цикл Trainer без изменения основного цикла обучения.
Они получают события обучения, эпох и батчей и могут сохранять артефакты, запрашивать
остановку обучения, собирать предсказания или отправлять уведомления.

Встроенные колбэки обычно создаются через реестр `CALLBACKS` в конфигурации
эксперимента. Напрямую объект удобно создавать в собственном цикле или тестах.
Примеры для конкретных задач находятся в разделе [Колбэки](../user-guide/callbacks.md).

## Контракт жизненного цикла

`BaseCallback` задаёт порядок точек расширения: `on_fit_start`,
`on_epoch_start`, повторяющиеся вызовы `on_batch_end`, затем
`on_epoch_end` и `on_fit_end`. Хуки получают экземпляр `Trainer`, а также
эпоху, глобальный шаг и агрегированные журналы, если это предусмотрено контрактом.
Реализация должна явно управлять побочными эффектами и использовать состояние Trainer.

::: chimera_ml.callbacks.base.BaseCallback

## Контрольные точки

`CheckpointCallback` наблюдает за одним скалярным журналируемым значением, записывает `last.pt`
и хранит лучшие контрольные точки в
`<log_path>/<experiment_name>/<run_name>/checkpoints`. Ключ реестра:
`checkpoint_callback`.

::: chimera_ml.callbacks.checkpoint_callback.CheckpointCallback

### Factory

::: chimera_ml.callbacks.checkpoint_callback.checkpoint_callback

## Оценка артефактов

`CollectPredictionsCallback` экспортирует кэшированные предсказания в CSV-артефакты,
а `PlotConfusionMatrixCallback` строит матрицу ошибок для классификации.
Оба колбэка используют логгер с поддержкой MLflow и включают кэширование предсказаний
в начале обучения.

::: chimera_ml.callbacks.collect_predictions_callback.CollectPredictionsCallback

::: chimera_ml.callbacks.collect_predictions_callback.collect_predictions_callback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.PlotConfusionMatrixCallback

::: chimera_ml.callbacks.plot_confusion_matrix_callback.plot_confusion_matrix_callback

## Управление обучением и observability

`EarlyStoppingCallback` устанавливает `trainer.stop_training` после заданного
терпения. `SweepTargetCallback` сохраняет лучшее контролируемое значение для перебора.
`SnapshotCallback` сохраняет снимки исходного кода и конфигурации, а
`TelegramNotifierCallback` отправляет итоговое сообщение с учётными данными из
переменных окружения.

::: chimera_ml.callbacks.early_stopping_callback.EarlyStoppingCallback

::: chimera_ml.callbacks.early_stopping_callback.early_stopping_callback

::: chimera_ml.callbacks.sweep_target_callback.SweepTargetCallback

::: chimera_ml.callbacks.sweep_target_callback.sweep_target_callback

::: chimera_ml.callbacks.snapshot_callback.SnapshotCallback

::: chimera_ml.callbacks.snapshot_callback.snapshot_callback

::: chimera_ml.callbacks.telegram_notifier_callback.TelegramNotifierCallback

::: chimera_ml.callbacks.telegram_notifier_callback.telegram_notifier_callback
