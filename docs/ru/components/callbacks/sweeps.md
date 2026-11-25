# Sweep objective callback — `SweepTargetCallback`

`SweepTargetCallback`, key `sweep_target_callback`, отслеживает лучший scalar для Optuna/grid sweep. Parameters: `monitor="val/loss"`, `mode="min"` или `"max"`. В `on_fit_start` reset-ит состояние, в `on_epoch_end` хранит `best_value`, `best_epoch`, `last_value`, `last_epoch`, `available_keys`. Training не останавливает и files не пишет.

Обычно sweep workflow добавляет callback автоматически; manual registration нужна только для собственного caller.

```yaml
- name: sweep_target_callback
  params: {monitor: val/mae, mode: min}
```
