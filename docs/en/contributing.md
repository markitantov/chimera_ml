# Contributing

Contributions should remain focused and include tests for behavior changes.
Start with the repository's [CONTRIBUTING.md](https://github.com/markitantov/chimera_ml/blob/main/CONTRIBUTING.md)
for the complete project policy.

## Local checks

Install the development and documentation groups, then run the same checks used
by CI:

```bash
poetry install --with dev,docs
poetry run pre-commit run --all-files
poetry run pytest -q
poetry run mkdocs build --strict --config-file mkdocs.yml
poetry run mkdocs build --strict --config-file mkdocs.ru.yml
```

For local preview, serve either language:

```bash
poetry run mkdocs serve --config-file mkdocs.yml
poetry run mkdocs serve --config-file mkdocs.ru.yml
```

Install the hooks once for local commits:

```bash
poetry run pre-commit install
```

## Documentation changes

The English version in `docs/en/` is the primary source. Keep the Russian
version in `docs/ru/` structurally symmetric so language and page switching stay
predictable. Do not translate Python identifiers, class and function names, CLI
commands, configuration keys, package names, import paths, or shell commands.
API docstrings are read directly from Python source and may remain in English.
