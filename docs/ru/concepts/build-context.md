# BuildContext

BuildContext — объект одного запуска, общий для CLI build stages. Он содержит
loaded config, stage (train или eval) и вложенное values для runtime metadata.

Компонент может публиковать metadata, не изменяя YAML:

~~~python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.num_classes", 7)
        context.set("data.feature_dim", 256)


def my_model(*, context=None):
    num_classes = context.get("data.num_classes")
    return MyModel(num_classes=num_classes)
~~~

После построения вызовите context.register(component). Если у component есть
describe_context(context), метод вызовет его и вернёт component.
register_many(components) делает то же для списка по порядку.

get(path, default) читает dotted paths, а set(path, value) создаёт
промежуточные dictionaries. CLI регистрирует datamodule до model, затем model,
loss, metrics, optimizer, scheduler и callbacks. Поэтому следующие компоненты
могут использовать metadata предыдущих.

Публичный contract — describe_context, get и set; не опирайтесь на private CLI state.
