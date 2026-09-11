# Configuration-driven архитектура

Train и eval используют YAML mapping. Framework проверяет верхнеуровневую
структуру, затем builders находят каждый name в registry и передают params
factory.

Минимальная структура:

~~~yaml
seed: 0

experiment_info:
  params:
    experiment_name: my-experiment

data:
  name: my_datamodule
  params: {}

model:
  name: my_model
  params: {}

train:
  params:
    epochs: 10

loss:
  name: mse_loss
  params: {}

optimizer:
  name: adamw_optimizer
  params:
    lr: 0.001
~~~

## Структурные правила

data, model, loss и optimizer должны быть mappings с непустым name. train
должен быть mapping, а его params передаются в TrainConfig. scheduler
необязателен, но при наличии должен иметь name. metrics, callbacks и logging
являются необязательными списками; каждый элемент должен иметь name.

experiment_info.params.experiment_name требуется для train и для
validate-config, если не указать --no-require-experiment-name.

## Named lists и runtime injection

Секции принимают mapping или list. logging обычно является списком:

~~~yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
~~~

Builders могут inject runtime objects: optimizer получает model, scheduler —
optimizer, а factory с явно объявленным context — текущий BuildContext.
При smart_inject=True передаются только явно объявленные параметры; одного
**kwargs недостаточно.

## Validation и execution

validate-config проверяет только framework-level structure. Unknown registry
keys, отсутствующие файлы, несовместимые plugin params и ошибки dataset
появляются при построении или запуске соответствующего workflow.

Поля и defaults описаны в [Configuration reference](../user-guide/configuration.md).
