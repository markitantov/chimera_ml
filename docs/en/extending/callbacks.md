# Custom callbacks

Subclass `BaseCallback` and implement only the lifecycle hooks you need: `on_fit_start(trainer)`, `on_epoch_start(trainer, epoch)`, `on_batch_end(trainer, global_step, logs)`, `on_epoch_end(trainer, epoch, logs)`, and `on_fit_end(trainer)`. Register a factory in `CALLBACKS`.

```python
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core import CALLBACKS

class LogLearningRate(BaseCallback):
    def on_epoch_end(self, trainer, epoch, logs):
        lr = trainer.optimizer.param_groups[0]["lr"]
        self._info(trainer, f"epoch={epoch} lr={lr}")

@CALLBACKS.register("log_learning_rate_callback")
def log_learning_rate_callback():
    return LogLearningRate()
```

Use `logs` for scalar epoch/batch values and public Trainer state for model, optimizer, scheduler, logger, cache, or `stop_training`. Do not confuse this lifecycle with inference `on_prediction`; that callback receives `(model, batch, output)` and is specific to prediction/XAI steps.
