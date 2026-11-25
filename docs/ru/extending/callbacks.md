# Пользовательские колбэки

Наследуйте `BaseCallback`, реализуйте нужные хуки `on_fit_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, `on_fit_end` и зарегистрируйте функцию построения в `CALLBACKS`.

```python
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core import CALLBACKS

class LogLearningRate(BaseCallback):
    def on_epoch_end(self, trainer, epoch, logs):
        self._info(trainer, f"epoch={epoch} lr={trainer.optimizer.param_groups[0]['lr']}")

@CALLBACKS.register("log_learning_rate_callback")
def log_learning_rate_callback(): return LogLearningRate()
```

`logs` предназначены для скалярных значений, а публичное состояние Trainer — для модели, оптимизатора, планировщика, кэша и `stop_training`. Это не инференсный `on_prediction`, который получает `(model, batch, output)` в шагах предсказания и XAI.
