# Быстрый старт

Это краткое руководство проверяет установленный CLI и показывает, где реальная
конфигурация эксперимента входит в рабочий процесс.

## Проверка окружения

После установки `chimera-ml` выполните:

```bash
chimera-ml doctor
chimera-ml --help
```

Команда `doctor` сообщает базовую информацию о Python, PyTorch, CUDA, MLflow и
реестрах/plugins, не запуская обучение.

## Проверка конфигурации

Установите один из plugin packages примеров, затем проверьте конфигурацию до
запуска эксперимента:

```bash
python -m pip install -e examples/va_estimation
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Проверка контролирует структуру YAML и обязательные поля эксперимента. Для
реального обучения в конфигурации примера всё ещё нужно указать пути к датасету
и выходным данным; эти пути, а также последующие команды train/eval описаны в
[README примера оценки VA](https://github.com/markitantov/chimera_ml/blob/main/examples/va_estimation/README.md).

## Использование контейнеров Python

Минимальное взаимодействие с библиотекой использует публичные контейнеры
`Batch` и `ModelOutput`:

```python
import torch

from chimera_ml import Batch, ModelOutput

batch = Batch(
    inputs={"audio": torch.zeros(1, 16000)},
    targets=None,
)
output = ModelOutput(preds=torch.zeros(1, 1))

assert batch.inputs["audio"].shape == (1, 16000)
assert output.preds.shape == (1, 1)
```

Модели принимают объекты `Batch` и возвращают объекты `ModelOutput`. В API
описаны эти контейнеры и `Registry`, который связывает встроенные компоненты и
компоненты плагинов.
