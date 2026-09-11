# Архитектура

Каталог src/chimera_ml разделён по runtime-ответственности:

- core: Batch, ModelOutput, ExperimentConfig и Registry;
- data: datamodule, нормализация loaders и collation;
- models, losses и metrics: граница модели и базовые training primitives;
- training: Trainer, builders, optimizer/scheduler factories и sweeps;
- callbacks и logging: lifecycle hooks, локальные logs и MLflow;
- inference: YAML-driven steps и sequential/DAG execution;
- utils: entry-point loading, seeds, sweep helpers и source snapshots.

CLI соединяет эти части, но не содержит task-specific datasets или models.
Они находятся в установленном plugin и регистрируют factories в тех же registry.

Граница модели явная: dataloader выдаёт Batch, model принимает Batch и
возвращает ModelOutput, loss и metrics получают оба объекта. Благодаря этому
plugin не зависит от реализации CLI.
