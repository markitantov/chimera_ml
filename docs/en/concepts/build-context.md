# BuildContext

`BuildContext` is a per-run object shared by the CLI build stages. It carries
the loaded config, a stage label (`train` or `eval`), and a nested `values`
mapping for runtime metadata.

Components can publish metadata without mutating the YAML:

```python
class MyDataModule:
    def describe_context(self, context):
        context.set("data.num_classes", 7)
        context.set("data.feature_dim", 256)


def my_model(*, context=None):
    num_classes = context.get("data.num_classes")
    return MyModel(num_classes=num_classes)
```

Call `context.register(component)` after building a component. The method calls
`describe_context(context)` when the component provides it and returns the
component. `register_many(components)` applies the same operation in order and
returns the original list.

`get(path, default)` reads dotted paths; `set(path, value)` creates missing
intermediate dictionaries. The CLI registers the datamodule before the model,
then registers the model, loss, metrics, optimizer, scheduler, and callbacks
as they are built. This ordering lets later components consume metadata
published by earlier components.

The public contract is metadata exchange through `describe_context`, `get`,
and `set`; a plugin should not depend on private CLI state.
