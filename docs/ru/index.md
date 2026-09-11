# Chimera ML

Chimera ML — configuration-driven Python framework для построения, обучения,
оценки и запуска одномодальных и мультимодальных PyTorch experiments. Runtime
предоставляет trainer, registries, callbacks, loggers и inference pipeline;
task-specific components могут находиться в установленном plugin package.

## С чего начать

1. [Начало работы](getting-started/index.md) — установите Chimera ML и
   проверьте первую configuration.
2. [Основные концепции](concepts/index.md) — configuration, registries,
   BuildContext и plugins.
3. [Руководство пользователя](user-guide/index.md) — training, evaluation,
   logging, inference, sweeps и CLI workflows.
4. [Практические примеры](tutorials/index.md) — реальные VA, ORAGEN и
   affective states projects.
5. [API Reference](api/index.md) и [CLI Reference](cli/reference.md) — точные
   interfaces и command options.
6. [Разработка](development/index.md) — contribution, tests, docs и release.
7. [Релизы](releases/index.md) — changelog и migration notes.

## Граница core API

Dataloader выдаёт Batch, model принимает Batch и возвращает ModelOutput, а
losses и metrics используют эти objects. Experiments описываются в YAML, а
components выбираются по registry key.

## Примеры репозитория

Это реальные plugin packages; для них нужны собственные data или model
artifacts:

- [VA estimation](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation)
- [ORAGEN](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen)
- [Affective states recognition](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition)

Установка published package:

~~~bash
python -m pip install chimera-ml
~~~

Текущий package поддерживает Python >=3.12,<3.13. Начните со страницы
[Установка](getting-started/installation.md).
