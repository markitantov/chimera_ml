# Loggers

The live `LOGGERS` registry contains `console_file_logger` and `mlflow_logger`.

## `console_file_logger`

Factory parameters: required `log_path`; `name="chimera"`, `experiment_name="chimera"`, `run_name="train"`, `log_file="train.log"`, `format="%(asctime)s:%(levelname)s:%(message)s"`, `console_level=INFO`, `file_level=INFO`, `file_mode="a"`, and `encoding="utf-8"`. It creates `<log_path>/<experiment_name>/<run_name>/<log_file>`, attaches console and file handlers, and exposes standard logging methods. The CLI injects experiment/run names when accepted.

```yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
      file_level: INFO
```

## `mlflow_logger`

Factory parameters: `tracking_uri=None`, `experiment_name="chimera"`, `run_name="train"`, and `config_path=None`. It lazily requires `mlflow`, starts/ends an MLflow run, logs scalar metrics and params, and supports file/text/bytes artifacts. The CLI can inject `config_path`, experiment name, and run name. `tracking_uri` may be local or a reachable tracking service.

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: demo
      run_name: baseline
```

Both loggers can be configured together. Callbacks that upload artifacts require an MLflow-capable logger; local console/file logging alone does not provide artifact storage. Keep workflow setup and UI instructions in [Training logging](../user-guide/logging.md).
