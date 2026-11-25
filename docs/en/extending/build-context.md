# BuildContext

`BuildContext` is per-experiment construction metadata, not an inference artifact store. It contains optional `config`, optional `stage`, and a nested `values` mapping.

```python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.feature_dim", 256)
        context.set("data.num_classes", 3)

def my_model(*, context=None):
    return MyModel(
        input_dim=context.get("data.feature_dim"),
        output_dim=context.get("data.num_classes"),
    )
```

`get(path, default=None)` reads dotted values; `set(path, value)` creates intermediate dictionaries. `register(component)` calls `component.describe_context(context)` when present and returns the same component; `register_many` applies this in order. The CLI builds/registers the DataModule before model, loss, metrics, optimizer, scheduler, and callbacks, so later factories can consume metadata.

Use BuildContext for construction-time dependency exchange. Use `InferenceContext.get_artifact/set_artifact` for runtime inference outputs; the two contexts are not interchangeable and BuildContext does not persist checkpoints or predictions.
