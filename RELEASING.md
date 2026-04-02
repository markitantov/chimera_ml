# Releasing `chimera-ml`

This project is already packaged with Poetry and can be installed as:

- `pip install chimera-ml`
- `poetry add chimera-ml`

once a version is published to PyPI.

## One-time setup

1. Create a project on PyPI with the name `chimera-ml`.
2. In GitHub repository settings, configure trusted publishing for PyPI:
   - Publisher: GitHub Actions
   - Workflow: `.github/workflows/publish.yml`
   - Environment: `pypi`
3. (Optional) Create a TestPyPI project to test first.

## Pre-release checklist

From repository root:

```bash
poetry install --with dev
poetry run pre-commit run --all-files
poetry run pytest -q
poetry build
```

Also update:

- `CHANGELOG.md` (`[Unreleased]` section)
- `README.md` if user-facing behavior changed

## Release flow

1. Bump version in `pyproject.toml`:

```bash
poetry version patch
# or: poetry version minor / poetry version major
```

2. Commit and push changes.
3. Create a GitHub Release (or push a tag and publish a release in UI).
4. Workflow `Publish` will:
   - build wheel + sdist,
   - verify that `CI` workflow succeeded for the same commit,
   - publish artifacts to PyPI via OIDC trusted publishing.

5. Verify install from PyPI in a clean environment:

```bash
python -m venv /tmp/chimera_ml_release_check
source /tmp/chimera_ml_release_check/bin/activate
python -m pip install -U pip
python -m pip install chimera-ml
chimera-ml --help
deactivate
```

Expected build artifacts in `dist/`:

- `chimera_ml-<version>-py3-none-any.whl`
- `chimera_ml-<version>.tar.gz`

## Manual run notes

`publish.yml` also supports `workflow_dispatch`.

- Default manual run (`publish=false`) only builds and validates distributions.
- To publish manually, start workflow with `publish=true`.
- Manual publish is blocked unless there is a successful `CI` run for the same commit SHA.
