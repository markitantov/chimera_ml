# Early stopping callback

`EarlyStoppingCallback`, registry key `early_stopping_callback`, monitors an exact scalar epoch-log key in `on_epoch_end` and sets `trainer.stop_training=True` after patience is exhausted.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `monitor` | `"val/loss"` | Log key |
| `mode` | `"min"` | Lower or higher is better |
| `patience` | `10` | Consecutive non-improving epochs; must be >= 1 |
| `min_delta` | `0.0` | Required improvement magnitude |

For `min`, improvement means `current < best - min_delta`; for `max`, `current > best + min_delta`. Missing keys warn and leave state unchanged. It does not save a model; pair it with `checkpoint_callback` when the best weights are required.

```yaml
- name: early_stopping_callback
  params: {monitor: val/loss, mode: min, patience: 5, min_delta: 0.0}
```
