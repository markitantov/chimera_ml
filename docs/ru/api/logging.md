# API журналирования

Логгеры предоставляют жизненный цикл, который использует `Trainer`: `start`,
скалярный `log_metrics` и `end`. Методы работы с артефактами являются необязательными
возможностями для колбэков. Логгер выбирается через реестр `LOGGERS`; см.
[Журналирование](../user-guide/logging.md) и [MLflow](../user-guide/mlflow.md).

## Контракт журналирования

::: chimera_ml.logging.base.BaseLogger

## Журналирование в консоль и файл

`ConsoleFileLogger` создаёт консольный/файловый логгер с меткой времени, удаляет
устаревшие обработчики во избежание дублирования вывода и предоставляет стандартные
методы журналирования. Ключ реестра: `console_file_logger`.

::: chimera_ml.logging.console_file_logger.ConsoleFileLogger

::: chimera_ml.logging.console_file_logger.console_file_logger

## Журналирование в MLflow

`MLflowLogger` запускает MLflow run, при необходимости записывает исходную конфигурацию,
отправляет метрики с явно заданным шагом и загружает файловые, текстовые или байтовые
артефакты. Ключ реестра: `mlflow_logger`.

::: chimera_ml.logging.mlflow_logger.MLflowLogger

::: chimera_ml.logging.mlflow_logger.mlflow_logger

## Утилиты именования

Вспомогательные функции создают локальные метки времени, короткие хэши и
понятные имена запусков для каталогов экспериментов и MLflow.

::: chimera_ml.logging.utils.local_datetime_tag

::: chimera_ml.logging.utils.short_hash

::: chimera_ml.logging.utils.generate_run_name
