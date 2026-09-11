# Первый эксперимент

В этом walkthrough используется реальный plugin package
examples/va_estimation. Это шаблон рабочего процесса, а не демо без
подготовки: для конфигураций нужны датасеты VA, признаки и локальные пути,
указанные в README примера.

## Структура проекта

~~~text
examples/va_estimation/
├── configs/
│   ├── multimodal_train.yaml
│   └── multimodal_test.yaml
├── src/
│   └── chimera_plugin.py
└── pyproject.toml
~~~

В pyproject.toml plugin объявлен как
va_estimation = "chimera_plugin:register". Функция register импортирует модули,
заполняющие registry Chimera ML.

## Конфигурация

Установите plugin и измените пути к датасету, признакам, аннотациям и
необязательным уведомлениям:

~~~bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config \
  --config-path examples/va_estimation/configs/multimodal_train.yaml
~~~

В конфиге компоненты data, model, loss и optimizer используют name и params;
metrics, callbacks и logging являются списками таких же элементов. См.
[Конфигурацию](../concepts/configuration.md).

## Обучение

~~~bash
chimera-ml train \
  --config-path examples/va_estimation/configs/multimodal_train.yaml
~~~

CLI создаёт BuildContext запуска, собирает компоненты из registry и вызывает
Trainer.fit. Callbacks checkpoint и snapshot записывают данные под
настроенным log root.

## Оценка

~~~bash
chimera-ml eval \
  --config-path examples/va_estimation/configs/multimodal_test.yaml \
  --checkpoint-path path/to/checkpoint.pt
~~~

Evaluation принимает checkpoint с model_state_dict или raw state dictionary и
строго загружает его в настроенную model. Затем оцениваются доступные train,
validation и test loader splits.

## Дальше

- Настройте [training](../user-guide/training.md), [metrics](../user-guide/metrics.md)
  и [callbacks](../user-guide/callbacks.md).
- Для inference и cache checkpoint см. [ORAGEN](../tutorials/oragen.md).
- Для расширения framework см. [Writing a plugin](../user-guide/plugins.md) и
  [руководство автора plugin](../development/plugins.md).
