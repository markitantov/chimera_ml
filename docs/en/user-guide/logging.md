# Logging

Use console_file_logger for Python logging to the console and a file:

~~~yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
      experiment_name: demo
      run_name: first-run
      log_file: train.log
      console_level: INFO
      file_level: INFO
~~~

The logger creates the experiment/run directory under log_path and exposes
run_name and log_path to the trainer/callback flow. The CLI injects generated
names when the factory signature accepts them.

Use [MLflow](mlflow.md) for tracked params, metrics, and artifacts. Both
logger entries can coexist.
