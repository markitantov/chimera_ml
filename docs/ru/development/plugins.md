# Руководство по созданию плагина

Это версия для разработчиков раздела [Создание плагина](../user-guide/plugins.md).

## Минимальный пакет

~~~text
my-plugin/
├── pyproject.toml
└── src/
    ├── chimera_plugin.py
    └── components.py
~~~

~~~toml
[project.entry-points."chimera_ml.plugins"]
my_plugin = "chimera_plugin:register"
~~~

~~~python
def register():
    import components
~~~

Функция построения регистрируется декоратором:

~~~python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim=128, context=None):
    return MyModel(hidden_dim=hidden_dim)
~~~

Компонент инференса регистрируется в `INFERENCE_STEPS`, возвращает объект с
`run(ctx)` и записывает результаты через `ctx.set_artifact`. Для контекста используйте
только объявленные метаданные выполнения.

## Тестирование и установка

~~~bash
python -m pip install -e my-plugin
chimera-ml plugins list
chimera-ml registry list --type models
~~~

Пишите тесты для регистрации, параметров функции построения и границы выполнения.
Эталонные реализации находятся в `examples/`.
