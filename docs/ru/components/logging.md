# Журналирование

В рабочем реестре `LOGGERS` зарегистрированы `console_file_logger` и `mlflow_logger`. Журналировщики записывают ход эксперимента, метрики и артефакты, чтобы результаты обучения можно было просмотреть и сопоставить между запусками.

## `console_file_logger`

Обязателен `log_path`; значения по умолчанию: `name="chimera"`, `experiment_name="chimera"`, `run_name="train"`, `log_file="train.log"`, формат `"%(asctime)s:%(levelname)s:%(message)s"`, уровень для консоли и файла `INFO`, `file_mode="a"`, `encoding="utf-8"`. Компонент создаёт `<log_path>/<experiment>/<run>/<log_file>` и предоставляет методы стандартного журналирования.

```yaml
logging:
  - name: console_file_logger
    params: {log_path: logs, file_level: INFO}
```

## `mlflow_logger`

Параметры: `tracking_uri=None`, `experiment_name="chimera"`, `run_name="train"`, `config_path=None`. Требуется отложенная зависимость `mlflow`; компонент создаёт запуск, записывает параметры и метрики, а также поддерживает файловые, текстовые и байтовые артефакты. CLI может внедрить имена конфигурации, эксперимента и запуска.

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: demo
      run_name: baseline
```

Оба журналировщика можно использовать вместе. Колбэки, загружающие артефакты, требуют журналировщик с поддержкой MLflow; инструкции по сценарию работы приведены в разделе [Журналирование обучения](../user-guide/logging.md).
