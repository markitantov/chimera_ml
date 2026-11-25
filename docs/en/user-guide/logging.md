# Training logging and MLflow

## Local console and file logging

`console_file_logger` is the local logger. It creates the run directory and writes the configured file while also sending messages to the console:

```yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
      log_file: train.log
      console_level: INFO
      file_level: INFO
      file_mode: a
      encoding: utf-8
```

The CLI injects `experiment_name` and the generated `run_name` when the factory accepts them. The file is therefore:

```text
logs/<experiment_name>/<run_name>/train.log
```

Trainer epoch messages include loss, sample count, and metric values. Batch logging is controlled by `train.params.log_every_steps` for MLflow step metrics; callbacks receive `train_step/loss` on every train batch.

## MLflow logger

`mlflow_logger` starts one MLflow run, logs training parameters and metrics, and accepts artifacts from callbacks:

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: va_estimation_fusion
      run_name: fusion
```

`tracking_uri` is optional. With a local SQLite URI, MLflow creates/uses the database path and its artifact storage relative to the configured MLflow behavior. A remote URI must point to a reachable MLflow tracking service. The CLI injects `config_path`, `experiment_name`, and `run_name` into the logger used by training.

The core imports MLflow lazily when this logger is constructed. If it is not installed, install the core runtime dependency or remove the MLflow logger. MLflow does not replace the local file logger; both can be configured together.

## What to inspect

- Local: `train.log`, `checkpoints/`, snapshots, and any plugin-written files below `logs/<experiment>/<run>/`.
- MLflow metrics: epoch metrics use split-prefixed keys such as `train/loss`, `val/loss`, and `val/mae`; step loss uses `train_step/loss`.
- MLflow artifacts: snapshot files use `snapshots`/`configs`; prediction callbacks use `predictions/<split>`; confusion matrices use `figures/<split>`.

A tracking URI alone does not make a run visible in a browser. Start the MLflow UI or connect to the configured server using the same tracking URI, for example for a local store:

```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

The `mlflow` executable belongs to the MLflow installation; this command is an optional inspection step, not part of `chimera-ml train`.
