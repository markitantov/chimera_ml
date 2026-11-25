# Колбэк цели перебора — `SweepTargetCallback`

`SweepTargetCallback`, ключ `sweep_target_callback`, отслеживает лучшее скалярное значение для перебора Optuna или полного перебора. Параметры: `monitor="val/loss"`, `mode="min"` или `"max"`. В `on_fit_start` сбрасывает состояние, а в `on_epoch_end` сохраняет `best_value`, `best_epoch`, `last_value`, `last_epoch`, `available_keys`. Обучение не останавливает и файлы не записывает.

Обычно сценарий перебора добавляет колбэк автоматически; ручная регистрация нужна только для собственного вызывающего кода.

```yaml
- name: sweep_target_callback
  params: {monitor: val/mae, mode: min}
```
