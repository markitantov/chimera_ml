# Реестры и строители

Публичные registries: `MODELS`, `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS`, `CALLBACKS`, `DATAMODULES`, `COLLATES`, `LOGGERS`, `INFERENCE_STEPS`.

```python
from chimera_ml.core import MODELS

@MODELS.register("my_model")
def my_model(*, hidden_dim: int = 128):
    return MyModel(hidden_dim=hidden_dim)
```

`register(key)` возвращает decorator и выдаёт `KeyError` при duplicate. `get` выдаёт unknown-key error с известными keys, `create` вызывает factory, `keys` возвращает sorted list. Builder lowercases names, требует mapping `params` и вызывает `factory(**kwargs)`.

```yaml
model:
  name: my_model
  params: {hidden_dim: 256}
```

Runtime injection: optimizer получает `model`, scheduler — `optimizer`, builders могут передать `context`, если factory явно его принимает. Built-ins импортируются через `chimera_ml.plugins.register_all`; installed entry point загружается в каждом CLI process. In-process decorator не виден отдельному CLI.

```bash
chimera-ml plugins list
chimera-ml registry list --type models
```
