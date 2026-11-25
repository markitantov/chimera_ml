# Evaluating a checkpoint

Standalone evaluation answers “how does this saved model perform on the configured splits?” Validation during `Trainer.fit` is part of each training epoch and informs callbacks; standalone `eval` performs no optimizer updates and can be run after training or on a separate machine with the required plugin/data.

## Command

```bash
chimera-ml eval   --config-path examples/va_estimation/configs/multimodal_test.yaml   --checkpoint-path logs/va_estimation_fusion/<run>/checkpoints/last.pt
```

Both options are required. `--with-features` additionally asks the Trainer to cache features from `ModelOutput.aux["features"]`:

```bash
chimera-ml eval -c test.yaml --checkpoint-path checkpoints/best.pt --with-features
```

## Loading and compatibility

The CLI rebuilds the DataModule, model, loss, metrics, optimizer, and callbacks from the config. It loads the checkpoint on CPU with `weights_only=True`. The file may be a callback payload containing `model_state_dict` or a raw state dictionary. `load_state_dict(strict=True)` requires the configured model architecture and parameter names to match exactly; changing modality names, dimensions, or constructor parameters usually makes the checkpoint incompatible.

## Splits, metrics, and cache

The CLI merges the datamodule's train, validation, and test loaders. A single loader is named by its split (`train`, `val`, or `test`); mapping keys are sanitized, and names that do not already begin with the split prefix receive one. Returned metrics therefore use keys such as `val/loss`, `val/mae`, or `test2/rmse`.

Evaluation invokes callback lifecycle hooks around one evaluation epoch. When `collect_cache` is enabled, predictions, targets, sample metadata, and optional features are cached per normalized split. Prediction and plotting callbacks can consume this cache. `--with-features` fails if the model does not expose `aux["features"]` and no direct feature extractor is supplied.

Evaluation artifacts depend on configured callbacks and loggers: local logs are under the configured run directory; MLflow receives metrics and callback-produced CSV/figure/config artifacts. Evaluation is not deployment inference and does not perform dataset-level XAI analysis.

## Troubleshooting evaluation

- Checkpoint errors usually mean the config does not describe the model that produced the file.
- Missing split output means the DataModule did not expose that dataset.
- Missing monitored metrics means the selected metric did not produce a scalar for that split.
- `--with-features` requires an `aux["features"]` tensor with the same batch dimension as predictions.
