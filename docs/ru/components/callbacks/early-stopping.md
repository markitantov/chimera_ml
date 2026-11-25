# Колбэк ранней остановки

`EarlyStoppingCallback`, key `early_stopping_callback`, в `on_epoch_end` наблюдает exact scalar log key и при исчерпании patience устанавливает `trainer.stop_training=True`.

| Parameter | Default | Значение |
| --- | --- | --- |
| `monitor` | `"val/loss"` | log key |
| `mode` | `"min"` | lower/higher is better |
| `patience` | `10` | consecutive non-improving epochs, >=1 |
| `min_delta` | `0.0` | required improvement |

Для `min`: `current < best - min_delta`, для `max`: `current > best + min_delta`. Missing key только warning. Callback не сохраняет модель; используйте вместе с checkpoint callback.

```yaml
- name: early_stopping_callback
  params: {monitor: val/loss, mode: min, patience: 5, min_delta: 0.0}
```
