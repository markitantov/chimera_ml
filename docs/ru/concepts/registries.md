# Реестры

Реестр сопоставляет стабильный строковый ключ с вызываемой функцией построения. В CHIMERA-ML
есть типы `datamodules`, `models`, `losses`, `metrics`, `optimizers`, `schedulers`,
`callbacks`, `collates`, `loggers` и `inference_steps`.

Встроенные компоненты регистрируются в `register_all()`. Плагины добавляют ключи
импортом модулей с декораторами реестра.

## Просмотр компонентов

~~~bash
chimera-ml registry list
chimera-ml registry list --type models
chimera-ml registry list --type inference_steps
~~~

Общая функция построения приводит ключи к нижнему регистру. Повторяющийся ключ —
ошибка, поэтому плагин не может молча заменить чужой компонент.

## Регистрация функции построения

~~~python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim: int = 128):
    return MyModel(hidden_dim=hidden_dim)
~~~

В YAML указывается ключ реестра, а не путь импорта Python:

~~~yaml
model:
  name: my_model
  params:
    hidden_dim: 256
~~~

См. [руководство по плагинам](../user-guide/plugins.md).
