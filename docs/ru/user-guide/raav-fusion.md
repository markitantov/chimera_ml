# RAAV fusion

RAAVFusionModel объединяет выровненные кадровые модальности и использует
отдельную контекстную модальность как узкое контекстное представление. Сначала
модель вычисляет учитывающие маски веса модальностей для каждого кадра, затем
обучаемые латентные векторы получают информацию из контекстной
последовательности. Каждый кадр обращается к этим латентным векторам,
получает контекстную поправку, а затем при необходимости проходит через
временной модуль.

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

- input_dims задаёт ширину признаков каждой кадровой и контекстной модальности;
  внутренние проекции переводят их в shared_dim.
- shared_dim — общая ширина механизма внимания, output_dim — ширина
  предсказания.
- frame_modalities — непустой кортеж имён выровненных временных модальностей.
- context_modality — имя независимого контекстного потока.
- num_latents задаёт размер узкого контекстного блока и должен быть положительным.
- gate_hidden — ширина скрытого слоя шлюза модальностей.
- temporal — необязательный Python-модуль, сохраняющий форму
  [B, T, shared_dim]; TransformerEncoder получает маску дополнения кадров.

## Входы и выходы

Входы кадров после сборки пакета имеют форму [B, T, D], причём T должен быть
одинаковым для присутствующих кадровых модальностей. У контекстного входа может
быть другая длина последовательности. Предсказания имеют форму
[B, T, output_dim], а невалидные позиции кадров обнуляются.

Приоритет имеет `<modality>_temporal_mask`, затем `<modality>_mask` и
совместимая sequence_mask. Отсутствующая кадровая модальность пропускается,
а отсутствие для отдельного образца учитывается через маски. Отсутствующий
или полностью замаскированный контекст даёт нулевые контекстные латенты и
нулевую контекстную поправку. Пустой контекст поддерживается. Пустая
последовательность кадров даёт пустую последовательность предсказаний; в
пакете должна присутствовать хотя бы одна кадровая модальность.

Полезные значения ModelOutput.aux: frame_weights, frame_mask, fused, context,
context_mask, context_latent_attention, frame_to_context_attention,
context_delta и refined.

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

Collate не выполняет автоматического выравнивания временных шкал. Подготовьте
кадровые потоки на общей временной шкале в наборе данных или заранее проверьте их
длины.

## Синтетический forward

Frame modalities должны иметь общий `T`, context может иметь другую длину:

```python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import RAAVFusionModel
model = RAAVFusionModel({"video": 4, "audio": 5, "text": 6}, shared_dim=8, output_dim=2, frame_modalities=("video", "audio"), context_modality="text").eval()
batch = Batch(inputs={"video": torch.randn(2, 3, 4), "audio": torch.randn(2, 3, 5), "text": torch.randn(2, 5, 6)}, targets=None)
output = model(batch)
assert output.preds.shape == (2, 3, 2)
assert output.aux["frame_mask"].shape == (2, 3)
```

Разная длина `audio`/`video` даёт `ValueError`, другая длина `text` разрешена.
