# Колбэк контрольных точек

`CheckpointCallback`, ключ `checkpoint_callback`, сохраняет состояние Trainer в `on_epoch_end`.

| Параметр | Значение по умолчанию | Назначение |
| --- | --- | --- |
| `log_path` | `"logs"` | Корневой каталог результатов |
| `experiment_name` | `"chimera"` | Каталог эксперимента |
| `run_name` | `"train"` | Каталог запуска |
| `monitor` | `"val/loss"` | Точный ключ журнала эпохи |
| `mode` | `"min"` | `min` или `max` |
| `save_top_k` | `1` | Число файлов под наблюдением; неположительное значение отключает удаление старых файлов |
| `save_last` | `True` | Обновляемый файл `last.pt` |
| `filename_template` | `epoch={epoch}_step={step}_{monitor}={value:.4f}.pt` | Поля epoch/step/monitor/value |

Файлы сохраняются в `<log_path>/<experiment>/<run>/checkpoints`. Полезная нагрузка содержит `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict` и `scheduler_state_dict`, если задан планировщик. Отсутствующий monitor вызывает предупреждение; колбэк не загружает контрольные точки и не возобновляет обучение.

```yaml
- name: checkpoint_callback
  params: {monitor: val/mae, mode: min, save_top_k: 2, save_last: true}
```

CLI внедряет `experiment_name` и `run_name` из `experiment_info.params`, если колбэк настроен; это значения времени выполнения.

Для загрузки используйте `torch.load`, `model.load_state_dict(payload["model_state_dict"])`, а при возобновлении — состояния оптимизатора и планировщика.
