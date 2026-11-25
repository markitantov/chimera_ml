# Быстрый старт

Это проверка без training, которую можно выполнить сразу после установки core. Dataset и checkpoint не нужны.

## 1. Проверить установку

```bash
chimera-ml doctor
chimera-ml --help
chimera-ml registry list --type models
chimera-ml registry list --type collates
```

В core install будут четыре built-in fusion model keys и два built-in collates. Plugin добавляет собственные keys.

## 2. Установить plugin и проверить YAML

Из корня репозитория:

```bash
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Проверка контролирует framework sections. Она не проверяет наличие paths, labels, credentials и optional model files, поэтому prerequisites нужно подготовить отдельно.

## 3. Проверить Python containers

```python
import torch
from chimera_ml import Batch, ModelOutput

batch = Batch(
    inputs={"audio": torch.zeros(2, 16000)},
    targets=torch.zeros(2, 1),
    masks={"audio_mask": torch.ones(2)},
    meta={"sample_meta": [{"id": "a"}, {"id": "b"}]},
)
output = ModelOutput(preds=torch.zeros(2, 1), aux={"features": torch.zeros(2, 8)})
assert batch.inputs["audio"].shape == (2, 16000)
assert output.preds.shape == (2, 1)
```

Model получает `Batch` и возвращает `ModelOutput`. Этот пример проверяет только containers.
