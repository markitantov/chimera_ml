# Contributing

Thanks for contributing to Chimera ML.

## Local setup

1. Install dependencies:

```bash
poetry install --with dev,docs
```

2. Install git hooks:

```bash
poetry run pre-commit install
```

3. Run quality checks before opening a PR:

```bash
poetry run pre-commit run --all-files
poetry run pytest -q
```

For documentation changes, build both sites strictly:

```bash
poetry run mkdocs build --strict -f mkdocs.en.yml
poetry run mkdocs build --strict -f mkdocs.ru.yml
```

## Pull requests

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update relevant English and Russian documentation when behavior or public usage changes.
- Update `README.md` and/or `CHANGELOG.md` when relevant.
- Preserve public component, registry-key, and configuration contracts; document changes to public interfaces and registry keys.
- Ensure CI is green before requesting review.
- If behavior changes for users, add a short note in `CHANGELOG.md` under `Unreleased`.

## Commit style

- Use clear, imperative commit messages.
- Prefer one logical change per commit.

## CLI changes

If you change `src/chimera_ml/cli.py`, also update:

- `tests/cli/test_cli.py`
- `README.md` CLI section (command syntax and examples)

## Reporting issues

When opening an issue, include:

- minimal reproduction steps,
- expected behavior,
- actual behavior,
- Python and dependency versions.
