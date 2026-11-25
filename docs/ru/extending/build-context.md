# BuildContext

`BuildContext` — хранилище метаданных построения одного эксперимента, а не хранилище артефактов инференса. Он содержит необязательные `config`, `stage` и вложенное `values`.

```python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.feature_dim", 256)
        context.set("data.num_classes", 3)

def my_model(*, context=None):
    return MyModel(context.get("data.feature_dim"), context.get("data.num_classes"))
```

`get(path, default=None)` читает значение по составному пути, а `set` создаёт промежуточные словари. `register(component)` вызывает `describe_context`, если этот метод есть, и возвращает компонент; `register_many` делает то же по порядку. CLI регистрирует DataModule до модели, функции потерь, метрик, оптимизатора, планировщика и колбэков.

Используйте BuildContext для метаданных на этапе построения. Для результатов инференса во время выполнения используйте `InferenceContext.get_artifact/set_artifact`; это разные контракты.
