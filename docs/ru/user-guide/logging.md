# Logging

console_file_logger пишет Python logs в console и file:

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

Logger создаёт experiment/run directory под log_path и предоставляет run_name и
log_path training/callback flow. Для tracked params, metrics и artifacts
используйте [MLflow](mlflow.md); оба logger entries могут работать вместе.
