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
poetry run mkdocs build --strict
```

Install the hooks once for local commits:

```bash
poetry run pre-commit install
```

## Documentation changes

Keep shared project guidance in `docs/`, and keep task-specific setup next to
the corresponding example. API pages should reference objects that exist in
`src/chimera_ml`; missing or incomplete docstrings are follow-up work rather
than a reason to make broad unrelated source changes.
