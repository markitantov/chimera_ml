# Callback catalog

A training callback subclasses `BaseCallback` (or implements the same hooks). Trainer invokes hooks in this order:

```text
on_fit_start
  for each epoch: on_epoch_start -> on_batch_end* -> on_epoch_end
on_fit_end
```

`on_batch_end` receives `global_step` and batch logs; `on_epoch_end` receives split-prefixed aggregate logs. Callbacks are built in YAML list order, and each hook is called in that order. The Trainer owns stop state, cached split outputs, model/optimizer/scheduler state, and logger references.

| Registry key | Main hook | Purpose |
| --- | --- | --- |
| `checkpoint_callback` | `on_epoch_end` | Save last and monitored top-k checkpoints |
| `early_stopping_callback` | `on_epoch_end` | Set `trainer.stop_training` after no improvement |
| `snapshot_callback` | `on_fit_start` | Archive source/config and optionally log artifacts |
| `collect_predictions_callback` | `on_fit_start`, `on_epoch_end` | Export cached split predictions as CSV to MLflow |
| `plot_confusion_matrix_callback` | `on_fit_start`, `on_epoch_end` | Plot classification confusion matrices |
| `sweep_target_callback` | `on_fit_start`, `on_epoch_end` | Track the best scalar sweep objective |
| `telegram_notifier_callback` | `on_fit_start`, `on_epoch_end`, `on_fit_end` | Send final run notification |

These are Trainer lifecycle callbacks. They are distinct from the inference `on_prediction(model, batch, output)` callback used by `xai_step`.

All keys are registered in `CALLBACKS`; inspect the current process with `chimera-ml registry list --type callbacks`. The detailed pages document constructor defaults and dependencies.
