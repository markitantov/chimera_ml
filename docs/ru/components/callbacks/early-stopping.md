# Колбэк ранней остановки

`EarlyStoppingCallback`, ключ `early_stopping_callback`, в `on_epoch_end` отслеживает точный скалярный ключ журнала и при исчерпании терпения устанавливает `trainer.stop_training=True`.

| Параметр | Значение по умолчанию | Значение |
| --- | --- | --- |
| `monitor` | `"val/loss"` | Ключ журнала |
| `mode` | `"min"` | Меньшее или большее значение считается лучшим |
| `patience` | `10` | Число последовательных эпох без улучшения, >=1 |
| `min_delta` | `0.0` | Минимально необходимое улучшение |

Для `min`: `current < best - min_delta`, для `max`: `current > best + min_delta`. Отсутствующий ключ вызывает только предупреждение. Колбэк не сохраняет модель; используйте его вместе с колбэком контрольных точек.

```yaml
- name: early_stopping_callback
  params: {monitor: val/loss, mode: min, patience: 5, min_delta: 0.0}
```
