# Архитектура на основе конфигурации

Команды `train` и `eval` используют отображение YAML. Фреймворк проверяет
верхнеуровневую структуру, затем функции построения находят каждое имя в реестре
и передают параметры соответствующей функции построения.

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


## Пример слияния и collate-функции для последовательностей

С помощью того же контракта реестра можно собрать модель слияния только по признакам
и collate-функцию для независимых последовательностей:

~~~yaml
model:
  name: raav_fusion_model
  params:
    input_dims: {video: 512, audio: 768, text: 1024}
    shared_dim: 256
    output_dim: 2
    frame_modalities: [video, audio]
    context_modalities: [text]

collate:
  name: multimodal_sequence_collate
  params:
    pad_sequences: true
~~~

Collate-функция создаёт отдельные маски наличия и времени для каждой модальности.
Модели слияния читают эти маски из Batch и не выравнивают независимые временные
оси автоматически.

Если эта верхнеуровневая секция присутствует, CLI `train` и `eval` разрешают её
через `build_collate()` и передают вызываемый объект функции построения DataModule как
`collate_fn`. Функция построения должна явно объявить этот именованный параметр. Без секции
`collate:` CLI сохраняет существующий DataModule по умолчанию и ничего не внедряет;
одновременная настройка верхнеуровневого `collate:` и
`data.params.collate_fn` считается ошибкой.

## Структурные правила

Разделы `data`, `model`, `loss` и `optimizer` должны быть отображениями с непустым `name`. Раздел `train` должен быть отображением, а его `params` передаются в TrainConfig. `scheduler` необязателен, но при наличии должен иметь `name`. `metrics`, `callbacks` и `logging` являются необязательными списками; каждый элемент должен иметь `name`.

`experiment_info.params.experiment_name` требуется для `train` и для
`validate-config`, если не указать `--no-require-experiment-name`.

## Именованные списки и внедрение во время выполнения

Секции принимают отображение или список. `logging` обычно является списком:

~~~yaml
logging:
  - name: console_file_logger
    params:
      log_path: logs
  - name: mlflow_logger
    params:
      tracking_uri: sqlite:///logs/mlflow.db
~~~

Функции построения могут внедрять объекты выполнения: оптимизатор получает модель,
планировщик — оптимизатор, а функция построения с явно объявленным `context` — текущий BuildContext.
При `smart_inject=True` передаются только явно объявленные параметры; одного
`**kwargs` недостаточно.

## Проверка и выполнение

`validate-config` проверяет только структуру уровня фреймворка. Неизвестные ключи
реестра, отсутствующие файлы, несовместимые параметры плагина и ошибки набора данных
выявляются при построении или запуске соответствующего сценария.

Поля и значения по умолчанию описаны в [справочнике конфигурации](../user-guide/configuration.md).
