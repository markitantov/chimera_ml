# Пользовательские колбэки

Наследуйте `BaseCallback`, реализуйте нужные hooks `on_fit_start`, `on_epoch_start`, `on_batch_end`, `on_epoch_end`, `on_fit_end` и зарегистрируйте factory в `CALLBACKS`.

```python
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core import CALLBACKS

class LogLearningRate(BaseCallback):
    def on_epoch_end(self, trainer, epoch, logs):
        self._info(trainer, f"epoch={epoch} lr={trainer.optimizer.param_groups[0]['lr']}")

@CALLBACKS.register("log_learning_rate_callback")
def log_learning_rate_callback(): return LogLearningRate()
```

`logs` предназначены для scalar values; public Trainer state — для model/optimizer/scheduler/cache/`stop_training`. Это не inference `on_prediction`, который получает `(model, batch, output)` для prediction/XAI steps.
