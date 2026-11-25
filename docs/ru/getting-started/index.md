# Начало работы

Читайте разделы по порядку. Core package предоставляет runtime и built-in registries; task-specific datasets и datamodules обычно приходят из plugin.

1. [Установка](installation.md): Python 3.12, CHIMERA-ML и проверка окружения.
2. [Быстрый старт](quickstart.md): CLI, registries и проверка реальной plugin-конфигурации без обучения.
3. [Первый эксперимент](first-experiment.md): plugin, данные, validation, training, artifacts и evaluation checkpoint.
4. [Основные понятия](essential-concepts.md): объекты training loop.

После этого переходите к [обзору обучения](../training/index.md).

## Обучение, оценка и инференс

- **Training** вызывает `Trainer.fit`, обновляет параметры модели и сохраняет checkpoints/logs. В train batch должны быть targets.
- **Evaluation** вызывает `Trainer.evaluate` с `train=False`, не делает optimizer updates и может сохранять predictions/features.
- **Inference** — отдельный `InferencePipeline` для входного файла. Он не заменяет evaluation и не использует loss/optimizer training loop.

`validate-config` проверяет форму YAML и обязательные framework sections. Он не скачивает datasets, не устанавливает plugins и не проверяет task-specific paths.
