# Logging и MLflow

## Local console/file logger

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

Logger создаёт `logs/<experiment_name>/<run_name>/train.log`; CLI inject-ит experiment и generated run names. Epoch logs содержат loss, sample count и metrics. `train.params.log_every_steps` задаёт interval step metrics для MLflow; callbacks получают `train_step/loss` на каждом train batch.

## MLflow logger

```yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: va_estimation_fusion
      run_name: fusion
```

Logger создаёт MLflow run, пишет params/metrics и принимает artifacts от callbacks. `tracking_uri` может быть local SQLite или reachable tracking service. MLflow импортируется при создании logger; при отсутствии dependency установите её или удалите logger. Local file и MLflow могут работать вместе.

Metrics имеют keys `train/loss`, `val/loss`, `val/mae`; step loss — `train_step/loss`. Artifacts используют `snapshots`/`configs`, `predictions/<split>`, `figures/<split>`.

Для local UI используйте optional command:

```bash
mlflow ui --backend-store-uri sqlite:///logs/mlflow.db
```

Tracking URI сам по себе не запускает browser UI и remote server.
