# TOMAR fusion

TOMARFusionModel использует одну modality как anchor и добавляет residual
corrections от остальных configured modalities. Для каждой correction отдельные
projection и MLP вычисляют gate по конкатенации anchor и pooled представления
дополнительной modality. Anchor и gated corrections проходят normalization
перед внутренним prediction layer.

## Параметры

~~~python
TOMARFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    anchor_modality: str,
    output_dim: int,
    gate_hidden: int = 64,
)
~~~

input_dims задаёт положительные feature widths и внутренние projections.
shared_dim — residual width, anchor_modality — обязательная modality,
output_dim — prediction width, gate_hidden — ширина каждого extra gate MLP.

## Входы, masks и выходы

Входы — feature-only tensors [B, D] или [B, T, D]. Temporal inputs
маскируются и pool-ятся в [B, shared_dim]; prediction имеет форму
[B, output_dim]. Dedicated temporal masks имеют приоритет над legacy presence
masks. Отсутствующая или полностью masked non-anchor modality не добавляет
residual. Anchor должен существовать и быть доступен для каждого sample; иначе
модель выбрасывает ValueError, а не создаёт скрыто неверный prediction.

ModelOutput.aux содержит anchor, по одному коэффициенту rho_`<modality>` и
residual delta_`<modality>` для каждой доступной non-anchor modality, а также
fused. Основной результат находится в preds.

## Python

~~~python
from chimera_ml.models import TOMARFusionModel

model = TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

## YAML

~~~yaml
model:
  name: tomar_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
    shared_dim: 256
    anchor_modality: text
    output_dim: 2
    gate_hidden: 64
~~~

Anchor является жёстким требованием. Другие modalities могут отсутствовать в
Batch.inputs, быть недоступны для отдельных samples или быть пустыми.
