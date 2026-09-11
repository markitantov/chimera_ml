# Архитектура для contributors

Начинайте трассировку изменений с src/chimera_ml:

- core задаёт containers, config loading и registries;
- training/builders переводит named YAML sections в runtime objects;
- training/trainer отвечает за optimization, evaluation, callbacks и logging;
- data нормализует loaders и создаёт masks;
- inference строит и выполняет step graphs;
- plugins.py импортирует built-ins и загружает entry points.

Tests организованы по подсистемам в tests/: core, data, models, losses,
metrics, callbacks, logging, inference, plugins, CLI и training. Examples —
отдельные packaged projects и часть integration surface.

При добавлении user-facing component обновите registration, tests, example/
config при необходимости, API reference и обе language trees.
