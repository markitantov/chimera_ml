# Registries

Registry сопоставляет стабильный строковый key с callable factory. В Chimera ML
есть типы datamodules, models, losses, metrics, optimizers, schedulers,
callbacks, collates, loggers и inference_steps.

Built-ins регистрируются в register_all(). Plugins добавляют keys импортом
модулей с registry decorators.

## Просмотр компонентов

~~~bash
chimera-ml registry list
chimera-ml registry list --type models
chimera-ml registry list --type inference_steps
~~~

Generic builder приводит keys к lowercase. Duplicate key — ошибка, поэтому
plugin не может молча заменить чужой component.

## Регистрация factory

~~~python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim: int = 128):
    return MyModel(hidden_dim=hidden_dim)
~~~

В YAML указывается key, а не Python import path:

~~~yaml
model:
  name: my_model
  params:
    hidden_dim: 256
~~~

См. [plugin guide](../user-guide/plugins.md).
