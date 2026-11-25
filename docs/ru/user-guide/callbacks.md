# Колбэки в сценарии обучения

Полный каталог колбэков и жизненного цикла находится в разделе [Колбэки](../components/callbacks/index.md); эта страница описывает настройку сценария работы.

Колбэки получают события жизненного цикла Trainer: `on_fit_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, `on_fit_end`. Монитор должен быть точным плоским ключом журнала.

## Контрольная точка, ранняя остановка и снимок

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

Пример предполагает метрику качества `mae` на выборке `val`. Контрольная точка создаётся в `logs/<experiment>/<run>/checkpoints`; снимок создаёт `code.zip` и копию конфигурации. CLI внедряет имена эксперимента и запуска, а также путь к конфигурации.

## Предсказания и матрица ошибок

Оба колбэка требуют журналировщик MLflow и включают кэш результатов оценки:

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

Колбэк предсказаний записывает CSV в `predictions/<split>`: для регрессии — `pred_*`/`target_*`, для классификации — `pred_class`, вероятности и целевое значение. Колбэк матрицы ошибок требует результатов, совместимых с классификацией, и записывает PDF в `figures/<split>`. Требуется `matplotlib`.

## Telegram и цель перебора

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

Notifier отправляет сообщение на `on_fit_end`, требует сетевого доступа и `requests`. Не записывайте токены в YAML. `sweep_target_callback` обычно автоматически добавляется для Optuna и отслеживает лучшее скалярное значение, но не останавливает обучение и не сохраняет файлы.

Контрольный список зависимостей: ключ мониторинга должен существовать; колбэкам предсказаний и графиков нужны кэшированные результаты для размеченных данных и MLflow; копирование конфигурации колбэком снимка требует пути к конфигурации CLI или явного `config_path` при прямом вызове API.
