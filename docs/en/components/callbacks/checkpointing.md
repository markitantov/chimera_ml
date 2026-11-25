# Checkpointing callback

`CheckpointCallback`, registry key `checkpoint_callback`, saves the current Trainer state in `on_epoch_end`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `log_path` | `"logs"` | Output root |
| `experiment_name` | `"chimera"` | Experiment directory |
| `run_name` | `"train"` | Run directory |
| `monitor` | `"val/loss"` | Exact epoch-log key |
| `mode` | `"min"` | `min` or `max` |
| `save_top_k` | `1` | Number of improving monitored files retained; non-positive disables pruning |
| `save_last` | `True` | Write rolling `last.pt` |
| `filename_template` | `epoch={epoch}_step={step}_{monitor}={value:.4f}.pt` | Format fields `epoch`, `step`, `monitor`, `value` |

Files go to `<log_path>/<experiment_name>/<run_name>/checkpoints`. The payload contains `epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`, and `scheduler_state_dict` when a scheduler exists. `last.pt` is written independently of the monitor. A missing monitor warns and skips the monitored checkpoint. The callback does not resume or load a checkpoint.

```yaml
- name: checkpoint_callback
  params:
    monitor: val/mae
    mode: min
    save_top_k: 2
    save_last: true
```

The CLI injects `experiment_name` and `run_name` from `experiment_info.params` when this callback is configured. YAML values can be overridden by those runtime values.

Load a saved payload explicitly with `torch.load`, `model.load_state_dict(payload["model_state_dict"])`, and corresponding optimizer/scheduler state when resuming. For evaluation/inference, use the model state compatible with the model constructor; the callback does not provide a loader.
