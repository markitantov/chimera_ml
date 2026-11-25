# Artifact callbacks

## `snapshot_callback` — `SnapshotCallback`

`SnapshotCallback` runs in `on_fit_start`. Parameters: `log_path="logs"`, `experiment_name="chimera"`, `run_name="train"`, `include=["src"]`, `save_code_zip=True`, `save_config=True`, and `config_path=None`. It creates `<log_path>/<experiment>/<run>/code.zip` from included paths and copies the config when `config_path` is set. With an MLflow logger it uploads code under `snapshots` and config under `configs`; upload failures warn. Direct API callers must provide `config_path` if they want a config copy.

```yaml
- name: snapshot_callback
  params:
    include: [src, pyproject.toml]
    save_code_zip: true
    save_config: true
```

The CLI injects snapshot `experiment_name`, `run_name`, and, when `save_config` is true, the current `config_path`. The prediction and confusion callbacks receive their cache/logger dependencies from Trainer rather than YAML injection.

## `collect_predictions_callback` — `CollectPredictionsCallback`

This callback enables `trainer.config.collect_cache=True` at fit start, then exports cached or freshly predicted split outputs at epoch end through `mlflow_logger`. Parameters: `splits=["val"]`, `artifact_path="predictions"`, `filename_template="preds_epoch_{epoch}.csv"`, `include_probs=True`, and `task="regression"` or `"classification"`. Split selectors include `train`, `val`, `test`, or exact named validation/test splits. Regression rows contain `pred_*` and `target_*`; classification rows contain `pred_class`, optional class probabilities, and a scalar target. Sample IDs are read from cached `sample_meta` where available. Without MLflow it returns without exporting.

```yaml
- name: collect_predictions_callback
  params:
    splits: [val, test]
    task: classification
    artifact_path: predictions
```

## `plot_confusion_matrix_callback` — `PlotConfusionMatrixCallback`

This callback also enables cache collection and requires an MLflow logger, classification-compatible cached predictions/targets, and the optional `matplotlib` dependency. Parameters are `splits=["val"]`, `class_names=None`, `artifact_path="figures"`, `filename_template="confusion_matrix_epoch_{epoch}.pdf"`, and `title_template="{split} Confusion Matrix (epoch {epoch})"`. It writes image/PDF artifacts per split. It cannot interpret regression outputs as classes.
