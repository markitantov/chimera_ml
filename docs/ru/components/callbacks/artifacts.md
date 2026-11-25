# Колбэки для артефактов

## `snapshot_callback` — `SnapshotCallback`

`SnapshotCallback`, ключ `snapshot_callback`, работает в `on_fit_start`. Параметры: `log_path="logs"`, `experiment_name="chimera"`, `run_name="train"`, `include=["src"]`, `save_code_zip=True`, `save_config=True`, `config_path=None`. Компонент создаёт `code.zip` и копирует конфигурацию, если задан `config_path`; при использовании MLflow загружает код в `snapshots`, а конфигурацию — в `configs`. При прямом вызове API путь к конфигурации нужно передать самостоятельно.

```yaml
- name: snapshot_callback
  params: {include: [src, pyproject.toml], save_code_zip: true, save_config: true}
```

CLI внедряет в колбэк снимка `experiment_name`, `run_name` и `config_path`, когда `save_config=true`; колбэки предсказаний и матрицы ошибок получают кэш и журналировщик от Trainer.

## `collect_predictions_callback` — `CollectPredictionsCallback`

Включает `trainer.config.collect_cache=True`, затем экспортирует кэшированные или только что рассчитанные результаты разделов через `mlflow_logger`. Параметры: `splits=["val"]`, `artifact_path="predictions"`, `filename_template="preds_epoch_{epoch}.csv"`, `include_probs=True`, `task="regression"` или `"classification"`. В качестве разделов можно выбрать `train`, `val`, `test` или точно именованный раздел. Строки регрессии содержат `pred_*`/`target_*`; строки классификации — `pred_class`, необязательные вероятности и целевое значение. Без MLflow экспорт не выполняется.

```yaml
- name: collect_predictions_callback
  params: {splits: [val, test], task: classification}
```

## `plot_confusion_matrix_callback` — `PlotConfusionMatrixCallback`

Колбэк включает кэширование результатов и требует наличия журналировщика MLflow, библиотеки `matplotlib`, а также предсказаний и целевых значений, соответствующих задаче классификации. Параметры: `splits=["val"]`, `class_names=None`, `artifact_path="figures"`, `filename_template="confusion_matrix_epoch_{epoch}.pdf"`, `title_template="{split} Confusion Matrix (epoch {epoch})"`. Он создаёт график для каждой выборки; регрессия не поддерживается.
