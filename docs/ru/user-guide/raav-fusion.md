# RAAV fusion

RAAVFusionModel объединяет выровненные frame modalities и использует отдельную
context modality как bottleneck. Сначала модель вычисляет mask-aware веса
modalities для каждого frame, затем trainable context latents attend к context
sequence. Каждый frame attend к этим context latents и получает context
residual; при необходимости после этого применяется temporal module.

## Параметры

~~~python
RAAVFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    frame_modalities: tuple[str, ...],
    context_modality: str,
    num_latents: int = 4,
    gate_hidden: int = 64,
    temporal: nn.Module | None = None,
)
~~~

- input_dims задаёт ширину признаков каждой frame и context modality; внутренние
  projections переводят их в shared_dim.
- shared_dim — общая attention width, output_dim — ширина prediction.
- frame_modalities — непустой tuple имён выровненных temporal modalities.
- context_modality — имя независимого context stream.
- num_latents задаёт context bottleneck и должен быть положительным.
- gate_hidden — ширина скрытого слоя frame-modality gate.
- temporal — необязательный Python module, сохраняющий форму
  [B, T, shared_dim]; TransformerEncoder получает frame padding mask.

## Входы и выходы

Frame inputs после collation имеют форму [B, T, D], причём T должен быть
одинаковым для присутствующих frame modalities. У context input может быть
другая temporal length. Predictions имеют форму [B, T, output_dim], а
невалидные frame positions обнуляются.

Приоритет имеет `<modality>_temporal_mask`, затем `<modality>_mask` и совместимый
sequence_mask. Отсутствующая frame modality пропускается, а per-sample
отсутствие учитывается через masks. Отсутствующий или полностью masked context
даёт нулевые context latents и нулевой context contribution. Пустой context
поддерживается. Пустая frame sequence даёт пустую prediction sequence; в
batch должна быть доступна хотя бы одна frame modality.

Полезные значения ModelOutput.aux: frame_weights, frame_mask, fused, context,
context_mask, context_latent_attention, frame_to_context_attention,
context_delta, refined.

## Python

~~~python
from chimera_ml.models import RAAVFusionModel

model = RAAVFusionModel(
    input_dims={"video": 512, "audio": 768, "text": 1024},
    shared_dim=256,
    output_dim=2,
    frame_modalities=("video", "audio"),
    context_modality="text",
    num_latents=4,
)
~~~

## YAML

~~~yaml
model:
  name: raav_fusion_model
  params:
    input_dims:
      video: 512
      audio: 768
      text: 1024
    shared_dim: 256
    output_dim: 2
    frame_modalities:
      - video
      - audio
    context_modality: text
    num_latents: 4
    gate_hidden: 64
~~~

Collate не выполняет автоматическое выравнивание timelines. Подготовьте frame
streams на общей timeline в dataset или заранее проверьте их длины.
