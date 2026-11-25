# BuildContext

`BuildContext` — metadata одного experiment construction, не inference artifact store. Он содержит optional `config`, `stage` и nested `values`.

```python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.feature_dim", 256)
        context.set("data.num_classes", 3)

def my_model(*, context=None):
    return MyModel(context.get("data.feature_dim"), context.get("data.num_classes"))
```

`get(path, default=None)` читает dotted value, `set` создаёт intermediate dictionaries. `register(component)` вызывает `describe_context`, если он есть, и возвращает component; `register_many` делает это по порядку. CLI регистрирует DataModule до model/loss/metrics/optimizer/scheduler/callbacks.

Используйте BuildContext для construction-time metadata. Для runtime inference outputs используйте `InferenceContext.get_artifact/set_artifact`; это разные контракты.
