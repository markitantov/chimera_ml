# TFS fusion

TFSFusionModel реализует feature-only вариант TripleFusion. Модель проецирует
признаки каждой modality в shared_dim, вычисляет направленные межмодальные
взаимодействия и выдаёт один prediction vector на sample. Для трёх modalities
с default groups используется иерархия в стиле MASAI:

1. шесть независимых направленных Transformer cross-attention blocks;
2. две тройки из направленных пар;
3. два reciprocal second-level cross-attention блока;
4. masked mean и population standard deviation pooling;
5. Linear -> BatchNorm -> Tanh, затем внутренний prediction layer.

Это иерархическая fusion-модель, а не простое усреднение pair outputs.

## Параметры

~~~python
TFSFusionModel(
    input_dims: dict[str, int],
    shared_dim: int,
    output_dim: int,
    groups: tuple[tuple[tuple[str, str], ...], ...] | None = None,
)
~~~

- input_dims — непустой mapping имени modality в ширину входных признаков.
  Для каждой modality создаётся внутренняя linear projection.
- shared_dim — общая ширина fusion-блоков.
- output_dim — ширина ModelOutput.preds.
- groups — необязательные группы направленных пар. По умолчанию каждая
  unordered pair представлена обоими направлениями. При custom groups или
  другом числе modalities используется generalized grouped hierarchy вместо
  трёхмодального MASAI-пути.

## Входы и выходы

Batch.inputs содержит feature-only tensors формы [B, D] или [B, T, D].
Векторы внутри модели нормализуются до одной temporal position.
ModelOutput.preds имеет форму [B, output_dim].

Маски читаются из Batch.masks, включая legacy nested masks. Выделенная
`<modality>_temporal_mask` формы [B, T] имеет приоритет; иначе `<modality>_mask`
может иметь форму [B] или [B, T]. Совместимый sequence_mask также
учитывается. Невалидные позиции не участвуют в attention и pooling, а
невалидные directed pairs обнуляются до second-level fusion. Modality может
отсутствовать в Batch.inputs; отсутствие для отдельного sample задаётся false
mask. Для fusion должна быть доступна хотя бы одна directed pair.

В aux находятся `pair_<source>-><target>`, `pair_valid_<source>-><target>`,
pair_weights, group_weights и fused. Основной результат находится в
ModelOutput.preds.

## Python

~~~python
from chimera_ml.models import TFSFusionModel

model = TFSFusionModel(
    input_dims={"text": 1024, "audio": 768, "video": 512},
    shared_dim=256,
    output_dim=2,
)
~~~

## YAML

~~~yaml
model:
  name: tfs_fusion_model
  params:
    input_dims:
      text: 1024
      audio: 768
      video: 512
    shared_dim: 256
    output_dim: 2
~~~

groups задаётся как список групп, внутри которых находятся пары из двух имён,
например [text, audio]. Пустые группы и повторяющиеся directed pairs
недопустимы. Если один tensor целиком отсутствует, может использоваться
generalized path; при наличии всех трёх tensors включается default hierarchy.
