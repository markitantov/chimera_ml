# Компоненты ядра

## Публичные импорты

Core экспортирует `Batch`, `ModelOutput`, `ExperimentConfig`, `load_yaml`, `Registry`, and the named registries. Data экспортирует `DataModule`, both collate classes/factories, and loader utilities; models экспортирует `BaseModel` and the four architecture classes/factories.

```python
from chimera_ml.core import Batch, ModelOutput, MODELS, Registry
from chimera_ml.data import DataModule, MaskingCollate, MultimodalSequenceCollate
from chimera_ml.models import BaseModel, TFSFusionModel
```

## `Batch`

`Batch` — публичный контейнер, который collate передаёт модели:

```python
Batch(
    inputs={"audio": audio_tensor, "video": video_tensor},
    targets=target_tensor_or_none,
    masks={"audio_mask": presence, "audio_temporal_mask": valid_positions},
    meta={"sample_meta": [...]},
)
```

- `inputs` связывает стабильные имена модальностей с tensors; batch dimension — axis 0, модальность может отсутствовать.
- `targets` — task target или `None` для unlabeled test/inference.
- `masks` — canonical flat mapping: `<modality>_mask` для presence, `<modality>_temporal_mask` для `[B,T]`, `sequence_mask` для shared padding.
- `meta` хранит IDs, lengths и application metadata, но не заменяет canonical masks.

`get_masks()` возвращает нормализованный mapping, а `get_masks("audio_mask")` — один mask. Для legacy compatibility читаются nested `Batch.masks["mask"]` и `meta["masks"]`; `{ "mask": {"audio": value} }` превращается в `audio_mask`. Модель сама определяет, какие модальности и маски поддерживает.

```python
# Unimodal labeled vector
Batch(inputs={"text": torch.randn(8, 768)}, targets=torch.randint(0, 3, (8,)))

# Multimodal temporal features и отсутствующий audio у одного sample
Batch(
    inputs={"audio": torch.randn(4, 12, 80), "video": torch.randn(4, 12, 512)},
    targets=torch.randn(4, 2),
    masks={"audio_mask": torch.tensor([1, 1, 0, 1]),
           "audio_temporal_mask": torch.ones(4, 12, dtype=torch.bool)},
)

# Unlabeled inference
Batch(inputs={"audio": torch.randn(1, 20, 80)}, targets=None)
```

Zero-filled missing input не равен настоящему нулевому сигналу; различие задают presence/temporal masks.

## `ModelOutput`

`ModelOutput(preds: torch.Tensor, aux: dict[str, torch.Tensor] | None = None)` — публичный результат модели. `preds` — logits для classification, logits для multi-label BCE или regression values; точная форма принадлежит задаче и модели. `aux` содержит optional tensors: fused features, attention weights, validity masks и diagnostics.

Trainer вычисляет loss из `output.preds` и `batch.targets`, а `aux` автоматически не оптимизирует. XAI копирует только явно запрошенные batch-aligned `aux` keys. Имена `aux` architecture-specific.

## Конфигурация и строители

Training YAML использует mapping `{"name": "mse_loss", "params": {"reduction": "mean"}}`. `build_from_registry` приводит name к lowercase, проверяет mapping `params`, объединяет injected runtime dependencies и вызывает `factory(**kwargs)`. `build_optimizer` injects `model`, scheduler — `optimizer`, а optional `context` передаётся только factory, которая его принимает.

Публичные builders: `build_model`, `build_loss`, `build_metrics`, `build_datamodule`, `build_collate`, `build_optimizer`, `build_scheduler`, `build_callbacks`, `build_logger`. Параметры factory не обязательно совпадают с параметрами underlying PyTorch class.
