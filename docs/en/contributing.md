# Contributing

Contributions should remain focused and include tests for behavior changes.
Start with the repository's [CONTRIBUTING.md](https://github.com/markitantov/chimera_ml/blob/main/CONTRIBUTING.md)
for the complete project policy.

## Documentation

Documentation lives in:

- `docs/en/` — English documentation
- `docs/ru/` — Russian documentation

To preview the English documentation locally:

```bash
poetry run mkdocs serve -f mkdocs.en.yml
```

To preview the Russian documentation:

```bash
poetry run mkdocs serve -f mkdocs.ru.yml
```

Before opening a pull request, verify both builds:

```bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
```
