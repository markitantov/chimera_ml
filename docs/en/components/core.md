# Core components

## Public imports

Core exports `Batch`, `ModelOutput`, `ExperimentConfig`, `load_yaml`, `Registry`, and the named registries. Data exports `DataModule`, both collate classes/factories, and loader utilities; models export `BaseModel` and the four architecture classes/factories.

```python
from chimera_ml.core import Batch, ModelOutput, MODELS, Registry
from chimera_ml.data import DataModule, MaskingCollate, MultimodalSequenceCollate
from chimera_ml.models import BaseModel, TFSFusionModel
```

## `Batch`

`Batch` is the public container passed from a collate/DataLoader to a model:

```python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor_or_none,
    masks={"audio_mask": presence, "audio_temporal_mask": valid_positions},
    meta={"sample_meta": [...]},
)
```

- `inputs` maps stable modality names to tensors. The batch dimension is axis 0. A modality may be absent from the mapping.
- `targets` is task-specific and can be `None` for unlabeled validation/test/inference data. Training requires labels where the configured loss needs them.
- `masks` is the canonical flat mask mapping. Common keys are `<modality>_mask` for sample presence, `<modality>_temporal_mask` for `[B,T]` positions, and `sequence_mask` for shared padded positions.
- `meta` carries IDs, lengths, and application metadata; it is not a replacement for canonical masks.

`get_masks()` returns the normalized flat mapping. `get_masks("audio_mask")` returns one entry. For compatibility it reads nested `Batch.masks["mask"]` and `meta["masks"]`, converting `{ "mask": {"audio": value} }` to `audio_mask`. Models still decide which keys they support; the container does not impose a universal modality schema.

Examples:

```python
# Unimodal vector, labeled
Batch(inputs={"text": torch.randn(8, 768)}, targets=torch.randint(0, 3, (8,)))

# Multimodal temporal features, with one missing sample-level modality
Batch(
    inputs={"audio": torch.randn(4, 12, 80), "video": torch.randn(4, 12, 512)},
    targets=torch.randn(4, 2),
    masks={"audio_mask": torch.tensor([1, 1, 0, 1]),
           "audio_temporal_mask": torch.ones(4, 12, dtype=torch.bool)},
)

# Unlabeled inference
Batch(inputs={"audio": torch.randn(1, 20, 80)}, targets=None)
```

A zero-filled missing input is not equivalent to a valid zero-valued signal; use the presence/temporal mask semantics emitted by the selected collate.

## `ModelOutput`

`ModelOutput(preds: torch.Tensor, aux: dict[str, torch.Tensor] | None = None)` is the public model result. `preds` is the task output consumed by the configured loss and metrics: logits for class-index classification, logits for multi-label BCE, or regression values. Its shape is task/model-specific. `aux` contains optional named tensors such as fused features, attention weights, validity masks, or per-modality diagnostics.

Trainer computes loss from `output.preds` and `batch.targets`; it does not automatically train on `aux`. XAI can select explicitly requested batch-aligned `aux` keys as diagnostics. Do not assume an architecture's `aux` names are shared by other models.

## Configuration and builders

Training YAML uses mappings such as `{"name": "mse_loss", "params": {"reduction": "mean"}}`. `build_from_registry` lowercases names, resolves the factory, validates that `params` is a mapping, merges injected runtime dependencies, and calls `factory(**kwargs)`. Runtime injection includes `model` for optimizer factories, `optimizer` for schedulers, and optional `context` when the factory declares it.

Public builders include `build_model`, `build_loss`, `build_metrics`, `build_datamodule`, `build_collate`, `build_optimizer`, `build_scheduler`, `build_callbacks`, and `build_logger`. Factory parameters are not automatically the same as underlying PyTorch constructor parameters; see each catalog page.
