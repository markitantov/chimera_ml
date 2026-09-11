# MLflow

mlflow_logger lazily imports MLflow, sets an optional tracking_uri and
experiment, starts a run, logs parameters and metrics, and ends the run after
training.

~~~yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: chimera
~~~

The trainer logs training parameters, optimizer learning rates, epoch metrics,
and batch loss at log_every_steps. Callbacks can log files, bytes, and text
through the logger interface. If MLflow is absent, the adapter raises an
actionable ModuleNotFoundError.

An URI such as sqlite:///logs/mlflow.db uses a local tracking store. Keep
secrets out of configs and use the tracking backend's authentication.
