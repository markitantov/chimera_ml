# Слияние RAAV

RAAVFusionModel выполняет покадровое слияние с уточнением контекстом.
Кадровые модальности объединяются на каждой временной позиции с помощью
обучаемых коэффициентов надёжности. Каждая контекстная модальность сворачивается,
переводится в общую размерность и используется для вычисления поправки для
каждого кадра. Результатом остаётся последовательность.

## Конструктор

~~~python
RAAVFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    frame_modalities: Sequence[str],
    context_modalities: Sequence[str],
    gate_hidden: int | None = None,
    dropout: float = 0.1,
    num_heads: int = 4,
)
~~~

frame_modalities и context_modalities должны быть непустыми и не должны
пересекаться. При gate_hidden=None используется shared_dim. Длины
контекстных последовательностей могут отличаться от длины кадровых
последовательностей. Кадровые потоки должны иметь одинаковую длину.

## Входы и выходы

Кадровые признаки имеют форму [B, T, D]; двумерные признаки считаются
признаками одной временной позиции. Предсказания имеют форму [B, T, output_dim].
Маски наличия, времени и последовательности определяют допустимые позиции.
Коэффициенты надёжности кадров возвращаются в aux["frame_weights"].

Дополнительно возвращаются frame_mask, fused, context_delta,
context_<name>, context_mask_<name> и context_attention_<name>. Отсутствующий
контекст даёт нулевую поправку. Должна присутствовать хотя бы одна кадровая
модальность.

## Пример

~~~python
import torch
from chimera_ml.core import Batch
from chimera_ml.models import RAAVFusionModel

model = RAAVFusionModel(
    {"video": 4, "audio": 5, "text": 6},
    shared_dim=8,
    output_dim=2,
    frame_modalities=("video", "audio"),
    context_modalities=("text",),
).eval()

batch = Batch(
    inputs={
        "video": torch.randn(2, 3, 4),
        "audio": torch.randn(2, 3, 5),
        "text": torch.randn(2, 5, 6),
    },
    targets=None,
)
output = model(batch)
assert output.preds.shape == (2, 3, 2)
~~~

Ключ в реестре: raav_fusion_model.
