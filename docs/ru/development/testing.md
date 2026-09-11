# Testing

Все tests:

~~~bash
poetry run pytest -q
~~~

Набор покрывает registry/config, loaders/masks, trainer integrations,
non-finite guards, callbacks, logging, inference graphs/checkpoint resolution,
CLI, plugin discovery и grid/Optuna sweeps.

Для сфокусированного изменения:

~~~bash
poetry run pytest tests/inference tests/cli -q
poetry run pytest tests/training/test_sweep.py -q
~~~

Даже documentation-only changes должны проходить оба strict MkDocs builds:
mkdocstrings импортирует package и разрешает API references.
