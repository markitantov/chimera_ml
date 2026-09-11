# MLflow

mlflow_logger лениво импортирует MLflow, задаёт tracking_uri и experiment,
запускает run, пишет params/metrics и завершает run после training.

~~~yaml
logging:
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
      experiment_name: chimera
~~~

Trainer пишет training params, optimizer learning rates, epoch metrics и
batch loss с интервалом log_every_steps. Callbacks могут отправлять files,
bytes и text через logger interface. Если MLflow отсутствует, возникает
понятный ModuleNotFoundError.

URI sqlite:///logs/mlflow.db использует local tracking store. Secrets
храните в backend authentication, не в configs.
