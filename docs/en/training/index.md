# Training overview

CHIMERA-ML training is a configuration-driven `Trainer.fit` workflow:

```text
Dataset -> DataModule -> DataLoader -> Collate -> Batch -> Model
        -> ModelOutput -> Loss/Metrics -> Optimizer -> Trainer
        -> Callbacks/Loggers -> Checkpoints
                           \-> Scheduler
```

## Ordered workflow

1. A plugin dataset reads one sample with one or more modality inputs, optional targets, and metadata.
2. A DataModule builds train, validation, and test DataLoaders. It may expose named multiple loaders.
3. A collate converts a list of sample mappings to one `Batch`, padding sequences and emitting masks when needed.
4. The Trainer moves the batch to the resolved device and calls `model(batch)`.
5. For labeled batches, the loss consumes `ModelOutput` and `Batch`; metrics update from the same pair.
6. Training backpropagates the loss, optionally clips gradients, steps the optimizer, and optionally steps the scheduler.
7. Validation runs after each training epoch. Callbacks receive logs such as `train/loss`, `val/loss`, and metric keys.
8. Loggers persist text, metrics, and artifacts. Checkpoint callbacks write model state under the run directory.

The core supplies `Batch`, `ModelOutput`, `DataModule`, built-in collates, Trainer, registry builders, built-in components, and CLI commands. The project supplies task data and usually registers a DataModule, model, loss, and metrics. The exact plugin factory signatures are authoritative for plugin-specific parameters.

## Start the workflow

```bash
chimera-ml validate-config --config-path config.yaml
chimera-ml train --config-path config.yaml
chimera-ml eval --config-path config.yaml --checkpoint-path logs/.../checkpoints/last.pt
```

Training and evaluation share component construction, but only training performs optimizer updates. Deployment-style inference is a separate pipeline; see the Inference tab.
