# Components overview

CHIMERA-ML separates stable public contracts from registered factories. A contract such as `BaseModel`, `Batch`, or `BaseMetric` defines how the Trainer calls a component; a registry key makes a factory constructible from YAML. Dataset implementations and task-specific prediction steps normally live in plugins.

| Category | Purpose | Public abstraction | Construction |
| --- | --- | --- | --- |
| Core | Exchange batches, outputs, config, and build metadata | `Batch`, `ModelOutput`, `ExperimentConfig`, `BuildContext` | Python imports and builders |
| Data | Turn dataset samples into loaders and batches | `DataModule`, `MaskingCollate`, `MultimodalSequenceCollate` | `DATAMODULES` / `COLLATES` |
| Models | Convert `Batch` to `ModelOutput` | `BaseModel` | `MODELS` or Python |
| Training | Compute loss/metrics and update parameters | `BaseLoss`, `BaseMetric`, optimizer/scheduler factories | `LOSSES`, `METRICS`, `OPTIMIZERS`, `SCHEDULERS` |
| Callbacks | Observe Trainer lifecycle and create artifacts/state | `BaseCallback` | `CALLBACKS` |
| Logging | Emit console/file metrics or MLflow artifacts | `BaseLogger` and logger-compatible objects | `LOGGERS` |
| Inference | Execute application steps and XAI | `InferenceContext`, step protocol | `INFERENCE_STEPS` |

A training run usually follows `Dataset -> DataModule -> DataLoader -> collate -> Batch -> BaseModel -> ModelOutput -> loss/metrics -> optimizer/scheduler`, while callbacks and loggers observe the Trainer. Inference uses a separate `InferenceContext` and step graph; it does not reuse `BuildContext` as an artifact store.

## Catalog

- [Core](core.md): `Batch`, `ModelOutput`, configuration, and builders.
- [Data](data.md): dataset responsibilities, DataModule, and both collates.
- [Models](models.md): model contract and the four built-in fusion architectures.
- [Training components](training.md): all built-in losses, metrics, optimizers, and schedulers.
- [Callbacks](callbacks/index.md): lifecycle and every registered callback.
- [Loggers](logging.md): console/file and MLflow logging.

Use `chimera-ml registry list` to inspect the components available in the current process. Plugin keys appear only after the plugin has been installed and discovered.
