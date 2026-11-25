# Registries and builders

The public registries are `MODELS`, `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS`, `CALLBACKS`, `DATAMODULES`, `COLLATES`, `LOGGERS`, and `INFERENCE_STEPS`.

```python
from chimera_ml.core import MODELS

@MODELS.register("my_model")
def my_model(*, input_dim: int, output_dim: int = 1):
    return MyModel(input_dim, output_dim)
```

`Registry.register(key)` returns a decorator and raises `KeyError` for a duplicate key. `get(key)` raises an error listing known keys; `create(key, **kwargs)` invokes the factory; `keys()` returns sorted keys. Generic builders lowercase names, require a mapping `params`, and call `factory(**params)`.

```yaml
model:
  name: my_model
  params: {input_dim: 32, output_dim: 2}
```

Builders may inject runtime dependencies: `build_optimizer` injects `model`, `build_scheduler` injects `optimizer`, and `build_*` functions can inject `context` when the factory signature opts in. With `smart_inject=True`, only explicitly named parameters receive injected values. Built-ins are imported by `chimera_ml.plugins.register_all`; installed plugin entry points are loaded in every CLI process. In-process decorator registration alone is not visible to a separate CLI process.

Verify both discovery and keys:

```bash
chimera-ml plugins list
chimera-ml registry list --type models
```
