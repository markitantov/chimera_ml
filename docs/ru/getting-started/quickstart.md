# Быстрый старт

Это проверка без обучения, которую можно выполнить сразу после установки ядра. Датасет и контрольная точка не нужны.

## 1. Проверить установку

```bash
chimera-ml doctor
chimera-ml --help
chimera-ml registry list --type models
chimera-ml registry list --type collates
```

В установленном ядре доступны четыре встроенных ключа моделей слияния и две встроенные функции collate. Плагин добавляет собственные ключи.

## 2. Установить плагин и проверить YAML

Из корня репозитория:

```bash
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Проверка контролирует разделы фреймворка. Она не проверяет наличие путей, меток, учётных данных и необязательных файлов модели, поэтому предварительные условия нужно подготовить отдельно.

## 3. Проверить контейнеры Python

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

Модель получает `Batch` и возвращает `ModelOutput`. Этот пример проверяет только контейнеры данных.
