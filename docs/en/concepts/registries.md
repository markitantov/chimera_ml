# Registries

A `Registry` maps a stable string key to a callable factory. Chimera ML has
these registry types: `datamodules`, `models`, `losses`, `metrics`,
`optimizers`, `schedulers`, `callbacks`, `collates`, `loggers`, and
`inference_steps`.

Built-ins register during `chimera_ml.plugins.register_all()`. Plugins add
keys by importing modules containing the registry decorators.

## Inspect registered components

```bash
chimera-ml registry list
chimera-ml registry list --type models
chimera-ml registry list --type inference_steps
```

Keys are normalized to lowercase by the generic builder. A duplicate key is an
error, which prevents a plugin from silently replacing another component.

## Register a factory

```python
from chimera_ml.core import MODELS


@MODELS.register("my_model")
def my_model(*, hidden_dim: int = 128):
    return MyModel(hidden_dim=hidden_dim)
```

The YAML then refers to the key, not the Python import path:

```yaml
model:
  name: my_model
  params:
    hidden_dim: 256
```

Use the [plugin guide](../user-guide/plugins.md) for packaging and discovery.
