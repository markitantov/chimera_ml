# Chimera ML

Chimera ML (Cross-modal Hierarchical Merging of Embeddings and Representations)
— лёгкий фреймворк для обучения и оценки настраиваемых одномодальных и
мультимодальных моделей.

Пакет предоставляет инфраструктуру времени выполнения — обучение, логирование,
колбэки, реестры и конвейеры инференса, — а компоненты, специфичные для задачи,
могут находиться во внешних plugin packages. Эксперименты задаются YAML-файлами
конфигурации и CLI `chimera-ml`.

## Установка

Chimera ML поддерживает Python `>=3.12,<3.13`:

```bash
pip install chimera-ml
```

## Быстрый старт

Проверьте установленное окружение и доступные компоненты времени выполнения:

```bash
chimera-ml doctor
chimera-ml registry list --type models
```

Пакет также предоставляет основные контейнеры данных, используемые моделями,
функциями потерь и метриками:

```python
import torch

from chimera_ml import Batch, ModelOutput

batch = Batch(inputs={"features": torch.zeros(2, 4)}, targets=None)
output = ModelOutput(preds=torch.zeros(2, 1))

print(batch.inputs["features"].shape)
print(output.preds.shape)
```

Для полноценного эксперимента установите plugin package, специфичный для задачи,
и проверьте один из YAML-файлов конфигурации репозитория. На странице [Быстрый
старт](getting-started/quickstart.md) описан этот процесс; наличие датасета или
чекпойнта заранее не предполагается.

## Карта документации

- [Установка](getting-started/installation.md) — поддерживаемая версия Python,
  установка пакета и настройка разработки.
- [Быстрый старт](getting-started/quickstart.md) — первые проверки CLI и рабочий
  процесс с plugin packages.
- [Руководства](guides/index.md) — взаимосвязь CLI, реестров, плагинов и YAML-конфигураций.
- [API](api/index.md) — автоматически сгенерированный reference для основной
  публичной части API.
- [Участие в разработке](contributing.md) — локальные проверки и рекомендации
  для pull request.

Более подробные запускаемые примеры находятся в репозитории:
[оценка VA](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation),
[ORAGEN](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen) и
[распознавание аффективных состояний](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition).

Исходный код и задачи доступны на
[GitHub](https://github.com/markitantov/chimera_ml).
