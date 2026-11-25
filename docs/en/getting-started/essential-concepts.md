# Essential concepts

## The objects in a training run

- **Dataset** returns one task sample. The core does not prescribe the storage format; a plugin dataset decides how paths, features, labels, and metadata are read.
- **DataModule** owns dataset construction and exposes `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`. A split may be one loader, a named mapping, or a sequence of loaders.
- **DataLoader** batches samples and calls a collate function.
- **Collate** converts sample mappings into a `Batch`. Built-ins are `masking_collate` and `multimodal_sequence_collate`.
- **Batch** contains `inputs`, optional `targets`, canonical `masks`, and optional `meta`.
- **Model** consumes `Batch` and returns `ModelOutput`.
- **ModelOutput** contains `preds` and optional tensor `aux` values such as features.
- **Loss** converts the model output and labeled batch into a scalar loss.
- **Metrics** accumulate predictions and targets across an epoch and return named scalars.
- **Optimizer** updates model parameters after backpropagation.
- **Scheduler** changes optimizer state either after each update or once per epoch.
- **Trainer** coordinates device placement, forward/loss/backward, metrics, logging, callbacks, and checkpoint state.
- **Callbacks** observe lifecycle events and may save artifacts or request early stopping.
- **Loggers** write local logs or MLflow metrics/artifacts.
- **Checkpoint** is a serialized model/optimizer (and optional scheduler) state written by a callback.

The framework supplies the containers, built-in models/losses/metrics/collates/callbacks/loggers, builders, and Trainer. A task project normally supplies the dataset, DataModule, and task-specific model/loss/metrics through a plugin.

## YAML and registries

A component is selected with `name` and configured with `params`:

```yaml
model:
  name: tfs_fusion_model
  params:
    input_dims: {audio: 128, video: 256}
    shared_dim: 64
    output_dim: 2
```

The key must exist in the corresponding registry. Inspect the current process rather than relying on a package from another environment:

```bash
chimera-ml registry list --type models
chimera-ml registry list --type datamodules
```

Plugins are discovered through the `chimera_ml.plugins` entry-point group during import. `plugins list` shows discovery; `registry list` shows the keys registered after discovery.
