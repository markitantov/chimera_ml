# API logging

Loggers предоставляют lifecycle, который использует \`Trainer\`: \`start\`,
scalar \`log_metrics\` и \`end\`. Artifact methods — optional capabilities для
callbacks. Logger выбирается через \`LOGGERS\` registry; см. [Logging](../user-guide/logging.md)
и [MLflow](../user-guide/mlflow.md).

## Logger contract

::: chimera_ml.logging.base.BaseLogger

## Console и file logging

\`ConsoleFileLogger\` создаёт timestamped console/file logger, удаляет stale
handlers для предотвращения duplicate output и предоставляет standard logging
methods. Registry key: \`console_file_logger\`.

::: chimera_ml.logging.console_file_logger.ConsoleFileLogger

::: chimera_ml.logging.console_file_logger.console_file_logger

## MLflow logging

\`MLflowLogger\` запускает MLflow run, optional logs source config, отправляет
metrics с explicit step и загружает file, text или byte artifacts. Registry key:
\`mlflow_logger\`.

::: chimera_ml.logging.mlflow_logger.MLflowLogger

::: chimera_ml.logging.mlflow_logger.mlflow_logger

## Naming utilities

Helpers создают local timestamp tags, short hashes и human-readable run names
для experiment directories и MLflow.

::: chimera_ml.logging.utils.local_datetime_tag

::: chimera_ml.logging.utils.short_hash

::: chimera_ml.logging.utils.generate_run_name
