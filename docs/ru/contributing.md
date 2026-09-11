# Участие в разработке

Изменения должны быть сфокусированными и сопровождаться тестами для изменений
поведения. Полная политика проекта приведена в [CONTRIBUTING.md](https://github.com/markitantov/chimera_ml/blob/main/CONTRIBUTING.md).

## Локальные проверки

Установите группы разработки и документации, затем запустите те же проверки, что
и CI:

```bash
poetry install --with dev,docs
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run mkdocs build --strict --config-file mkdocs.yml
poetry run mkdocs build --strict --config-file mkdocs.ru.yml
```

Для локального просмотра можно запустить любой язык:

```bash
poetry run mkdocs serve --config-file mkdocs.yml
poetry run mkdocs serve --config-file mkdocs.ru.yml
```

Установите hooks один раз для локальных коммитов:

```bash
poetry run pre-commit install
```

## Изменения документации

Английская версия в `docs/en/` является основной. Русская версия в `docs/ru/`
должна сохранять ту же структуру относительных путей. Не переводите Python
identifiers, имена классов и функций, CLI commands, configuration keys, package
names, import paths и shell commands. API docstrings берутся непосредственно из
исходников Python и могут оставаться на английском языке.
