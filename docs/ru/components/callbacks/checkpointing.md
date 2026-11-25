# Колбэк контрольных точек

`CheckpointCallback`, key `checkpoint_callback`, сохраняет Trainer state в `on_epoch_end`.

| Parameter | Default | Назначение |
| --- | --- | --- |
| `log_path` | `"logs"` | output root |
| `experiment_name` | `"chimera"` | experiment directory |
| `run_name` | `"train"` | run directory |
| `monitor` | `"val/loss"` | exact epoch-log key |
| `mode` | `"min"` | `min` или `max` |
| `save_top_k` | `1` | число monitored files; non-positive отключает pruning |
| `save_last` | `True` | rolling `last.pt` |
| `filename_template` | `epoch={epoch}_step={step}_{monitor}={value:.4f}.pt` | fields epoch/step/monitor/value |

Файлы: `<log_path>/<experiment>/<run>/checkpoints`. Payload содержит `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict` и `scheduler_state_dict`, если scheduler есть. Missing monitor вызывает warning; callback не загружает и не resume checkpoint.

```yaml
- name: checkpoint_callback
  params: {monitor: val/mae, mode: min, save_top_k: 2, save_last: true}
```

CLI injects `experiment_name` и `run_name` из `experiment_info.params`, если callback настроен; это runtime values.

Для загрузки используйте `torch.load`, `model.load_state_dict(payload["model_state_dict"])` и state optimizer/scheduler при resume.
