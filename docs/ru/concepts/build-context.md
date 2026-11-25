# BuildContext

BuildContext — объект одного запуска, общий для этапов построения CLI. Он содержит
загруженную конфигурацию, stage (`train` или `eval`) и вложенное `values` для метаданных выполнения.

Компонент может публиковать метаданные, не изменяя YAML:

~~~python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.num_classes", 7)
        context.set("data.feature_dim", 256)


def my_model(*, context=None):
    num_classes = context.get("data.num_classes")
    return MyModel(num_classes=num_classes)
~~~

После построения вызовите `context.register(component)`. Если у компонента есть
`describe_context(context)`, метод вызовет его и вернёт компонент.
`register_many(components)` делает то же для списка по порядку.

`get(path, default)` читает значения по составным путям, а `set(path, value)` создаёт
промежуточные словари. CLI регистрирует DataModule до модели, затем модель, функцию
потерь, метрики, оптимизатор, планировщик и колбэки. Поэтому следующие компоненты
могут использовать метаданные предыдущих.

Публичный контракт — `describe_context`, `get` и `set`; не опирайтесь на внутреннее состояние CLI.
