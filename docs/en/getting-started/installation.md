# Installation

## Requirements

Chimera ML currently supports Python `>=3.12,<3.13`. The package declares
PyTorch `>=2.2,<3.0` and the other runtime dependencies in `pyproject.toml`.

## Install from PyPI

```bash
python -m pip install chimera-ml
```

The package installs the `chimera-ml` CLI. The underscore alias `chimera_ml`
is also available, but the documentation consistently uses the hyphenated
name.

## Install from source

For development, install the project and its development tools with Poetry:

```bash
poetry install --with dev,docs
```

The `docs` group contains MkDocs, Material for MkDocs, and the Python handler
for `mkdocstrings`. It is intentionally separate from the production runtime
dependencies.

## Install an example plugin

Task-specific datamodules, models, losses, metrics, and callbacks are commonly
provided by external plugins. From the repository root, an example plugin can
be installed in editable mode:

```bash
python -m pip install -e examples/va_estimation
```

Example plugin dependencies and dataset paths are documented in each example's
README. Installing a plugin is not required for the core CLI diagnostics or
the generated API reference.
