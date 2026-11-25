# Каталог callbacks

Training callback наследует `BaseCallback` или реализует те же hooks. Trainer вызывает их в порядке:

```text
on_fit_start
  для каждой эпохи: on_epoch_start → on_batch_end* → on_epoch_end
on_fit_end
```

`on_batch_end` получает `global_step` и batch logs; `on_epoch_end` — split-prefixed aggregate logs. Callbacks строятся в порядке YAML list. Это lifecycle callbacks Trainer, а не inference `on_prediction(model, batch, output)`, который использует `xai_step`.

| Registry key | Hook | Назначение |
| --- | --- | --- |
| `checkpoint_callback` | `on_epoch_end` | last и monitored top-k checkpoints |
| `early_stopping_callback` | `on_epoch_end` | `trainer.stop_training` после отсутствия улучшения |
| `snapshot_callback` | `on_fit_start` | source/config snapshot |
| `collect_predictions_callback` | start/end | cached prediction CSV в MLflow |
| `plot_confusion_matrix_callback` | start/end | classification confusion figures |
| `sweep_target_callback` | start/end | best scalar sweep objective |
| `telegram_notifier_callback` | start/end | final Telegram notification |

Проверяйте registry: `chimera-ml registry list --type callbacks`.


## Пример confusion matrix

Callback предназначен только для classification. Он требует cache collection,
MLflow logger и optional dependency matplotlib.

~~~yaml
- name: plot_confusion_matrix_callback
  params:
    splits: [val]
    class_names: [negative, neutral, positive]
    artifact_path: figures
    filename_template: confusion_matrix_epoch_{epoch}.pdf
    title_template: "{split} Confusion Matrix (epoch {epoch})"
~~~

В on_fit_start callback включает collection, а в on_epoch_end пишет figure для
каждого выбранного split. Regression predictions он интерпретировать не может.
