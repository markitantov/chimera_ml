# Участие в разработке

Изменения должны быть сфокусированными и сопровождаться тестами для изменений
поведения. Полная политика проекта приведена в [CONTRIBUTING.md](https://github.com/markitantov/chimera_ml/blob/main/CONTRIBUTING.md).

## Документация

Документация находится в каталогах:

- `docs/en/` — документация на английском языке
- `docs/ru/` — документация на русском языке

Для локального просмотра английской документации:

```bash
poetry run mkdocs serve -f mkdocs.en.yml
```

Для локального просмотра русской документации:

```bash
poetry run mkdocs serve -f mkdocs.ru.yml
```

Перед открытием pull request проверьте обе сборки:

```bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
```
