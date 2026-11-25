# Реестры и функции построения

Публичные реестры: `MODELS`, `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS`, `CALLBACKS`, `DATAMODULES`, `COLLATES`, `LOGGERS`, `INFERENCE_STEPS`.

```python
from chimera_ml.core import MODELS

@MODELS.register("my_model")
def my_model(*, hidden_dim: int = 128):
    return MyModel(hidden_dim=hidden_dim)
```

`register(key)` возвращает декоратор и выдаёт `KeyError` при повторяющемся ключе. `get` выдаёт ошибку неизвестного ключа с перечнем известных ключей, `create` вызывает функцию построения, а `keys` возвращает отсортированный список. Функция построения приводит имена к нижнему регистру, требует отображение `params` и вызывает `factory(**kwargs)`.

```yaml
model:
  name: my_model
  params: {hidden_dim: 256}
```

Внедрение во время выполнения: оптимизатор получает `model`, планировщик — `optimizer`, а функции построения могут получить `context`, если явно принимают его. Встроенные компоненты импортируются через `chimera_ml.plugins.register_all`; установленная точка входа загружается в каждом процессе CLI. Декоратор, сработавший в одном процессе, не виден отдельному CLI.

```bash
chimera-ml plugins list
chimera-ml registry list --type models
```
