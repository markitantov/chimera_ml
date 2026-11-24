# Adaptive TFS-DCMMOE fusion

TFSDCMMOEFusionModel сохраняет иерархию TFS и добавляет входозависимые gates
над directed pair experts. Для каждой доступной pair вычисляется score, а
masked softmax исключает недоступные pairs до применения весов. Поэтому модель
может выбирать relations отдельно для каждого sample.

Для трёх modalities и default groups используются те же шесть directed
Transformer-блоков первого уровня и два reciprocal second-level блока, что и
в TFS. Обычный TFS использует бинарную доступность pair, а DCMMOE применяет
learned normalized pair weights, сохраняя нулевой вес для недоступных experts.

## Параметры

~~~python
TFSDCMMOEFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
    gate_hidden: int = 64,
)
~~~

input_dims, shared_dim, output_dim и groups имеют тот же смысл, что в TFS.
gate_hidden — ширина скрытого слоя pair-gating MLP. Размерности должны быть
положительными.

## Входы, masks и выходы

В Batch.inputs передаются feature-only tensors [B, D] или [B, T, D]; модель
сама проецирует их в shared_dim. Prediction имеет форму [B, output_dim].
Dedicated temporal masks имеют приоритет над legacy presence masks, а полностью
masked pairs получают нулевой вес. Отсутствующая modality и false mask для
отдельного sample не влияют на valid pair или second-level fusion. Нужна хотя
бы одна доступная directed pair.

ModelOutput.aux содержит pair representations и validity keys, а также
pair_weights, group_weights и fused. pair_weights показывает mask-aware
adaptive expert weights; task output находится в preds.

## Python

~~~python
from chimera_ml.models import TFSDCMMOEFusionModel

model = TFSDCMMOEFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
    gate_hidden=128,
)
~~~

## YAML

~~~yaml
model:
  name: tfs_dcmmmoe_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
    gate_hidden: 128
~~~

groups позволяет задать явную grouped hierarchy. При нестандартных groups или
менее чем трёх names используется generalized TFS path, но adaptive masked pair
weighting сохраняется.
