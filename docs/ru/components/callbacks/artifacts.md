# Колбэки для артефактов

## `snapshot_callback` — `SnapshotCallback`

`SnapshotCallback`, key `snapshot_callback`, работает в `on_fit_start`. Parameters: `log_path="logs"`, `experiment_name="chimera"`, `run_name="train"`, `include=["src"]`, `save_code_zip=True`, `save_config=True`, `config_path=None`. Создаёт `code.zip` и копирует config, если задан `config_path`; при MLflow загружает code в `snapshots`, config в `configs`. Direct API должен сам передать config path.

```yaml
- name: snapshot_callback
  params: {include: [src, pyproject.toml], save_code_zip: true, save_config: true}
```

CLI injects для snapshot `experiment_name`, `run_name` и `config_path`, когда `save_config=true`; prediction/confusion callbacks получают cache/logger от Trainer.

## `collect_predictions_callback` — `CollectPredictionsCallback`

Включает `trainer.config.collect_cache=True`, затем экспортирует cached или freshly predicted split outputs через `mlflow_logger`. Parameters: `splits=["val"]`, `artifact_path="predictions"`, `filename_template="preds_epoch_{epoch}.csv"`, `include_probs=True`, `task="regression"` или `"classification"`. Selectors: `train`, `val`, `test` или exact named split. Regression rows содержат `pred_*`/`target_*`; classification — `pred_class`, optional probabilities и target. Без MLflow export не выполняется.

```yaml
- name: collect_predictions_callback
  params: {splits: [val, test], task: classification}
```

## `plot_confusion_matrix_callback` — `PlotConfusionMatrixCallback`

Включает cache и требует MLflow, classification-compatible predictions/targets и `matplotlib`. Parameters: `splits=["val"]`, `class_names=None`, `artifact_path="figures"`, `filename_template="confusion_matrix_epoch_{epoch}.pdf"`, `title_template="{split} Confusion Matrix (epoch {epoch})"`. Создаёт figures per split; regression не поддерживается.
