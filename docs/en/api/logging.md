# Logging API

Loggers provide the lifecycle used by \`Trainer\`: \`start\`, scalar
\`log_metrics\`, and \`end\`. Artifact methods are optional capabilities used by
callbacks. Select a logger from the \`LOGGERS\` registry; see [Logging](../user-guide/logging.md)
and [MLflow](../user-guide/mlflow.md) for setup and operational guidance.

## Logger contract

::: chimera_ml.logging.base.BaseLogger

## Console and file logging

\`ConsoleFileLogger\` creates a timestamped console/file logger, removes stale
handlers to avoid duplicate output, and exposes standard logging methods.
Registry key: \`console_file_logger\`.

::: chimera_ml.logging.console_file_logger.ConsoleFileLogger

::: chimera_ml.logging.console_file_logger.console_file_logger

## MLflow logging

\`MLflowLogger\` starts an MLflow run, optionally logs the source config, sends
metrics with an explicit step, and uploads file, text, or byte artifacts.
Registry key: \`mlflow_logger\`.

::: chimera_ml.logging.mlflow_logger.MLflowLogger

::: chimera_ml.logging.mlflow_logger.mlflow_logger

## Naming utilities

These helpers produce local timestamp tags, short hashes, and human-readable
run names suitable for experiment directories and MLflow.

::: chimera_ml.logging.utils.local_datetime_tag

::: chimera_ml.logging.utils.short_hash

::: chimera_ml.logging.utils.generate_run_name
