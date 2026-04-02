# Contributing

Thanks for contributing to Chimera ML.

## Local setup

1. Install dependencies:

```bash
poetry install --with dev
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

## Pull requests

- Keep PRs focused and small when possible.
- Add or update tests for behavior changes.
- Update `README.md` and/or `CHANGELOG.md` when relevant.
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
