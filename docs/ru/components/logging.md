# Журналирование

В live `LOGGERS` registry находятся `console_file_logger` и `mlflow_logger`.

## `console_file_logger`

Required `log_path`; defaults: `name="chimera"`, `experiment_name="chimera"`, `run_name="train"`, `log_file="train.log"`, format `"%(asctime)s:%(levelname)s:%(message)s"`, console/file level `INFO`, `file_mode="a"`, `encoding="utf-8"`. Создаёт `<log_path>/<experiment>/<run>/<log_file>` и proxy standard logging methods.

```yaml
logging:
  - name: console_file_logger
    params: {log_path: logs, file_level: INFO}
```

## `mlflow_logger`

Parameters: `tracking_uri=None`, `experiment_name="chimera"`, `run_name="train"`, `config_path=None`. Требует lazy dependency `mlflow`, создаёт run, logs params/metrics и supports file/text/bytes artifacts. CLI может inject config/experiment/run names.

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: demo
      run_name: baseline
```

Оба logger можно использовать вместе. Callbacks, загружающие artifacts, требуют MLflow-capable logger; workflow instructions остаются в [Training logging](../user-guide/logging.md).
