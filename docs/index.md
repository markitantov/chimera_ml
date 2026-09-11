# Chimera ML

Chimera ML (Cross-modal Hierarchical Merging of Embeddings and Representations)
is a lightweight framework for training and evaluating configurable uni-modal
and multi-modal models.

The package provides runtime infrastructure—training, logging, callbacks,
registries, and inference pipelines—while task-specific components can live in
external plugin packages. Experiments are driven by YAML configuration files
and the `chimera-ml` CLI.

## Installation

Chimera ML supports Python `>=3.12,<3.13`:

```bash
pip install chimera-ml
```

## Quick start

Check the installed environment and available runtime components:

```bash
chimera-ml doctor
chimera-ml registry list --type models
```

The package also exposes the core data containers used by models, losses, and
metrics:

```python
import torch

from chimera_ml import Batch, ModelOutput

batch = Batch(inputs={"features": torch.zeros(2, 4)}, targets=None)
output = ModelOutput(preds=torch.zeros(2, 1))

print(batch.inputs["features"].shape)
print(output.preds.shape)
```

For a complete experiment, install a task-specific plugin and validate one of
the repository's YAML configurations. The [Quick start](getting-started/quickstart.md)
page explains this flow without assuming that datasets or model checkpoints are
already available.

## Documentation map

- [Installation](getting-started/installation.md) — supported Python version,
  package installation, and development setup.
- [Quick start](getting-started/quickstart.md) — first CLI checks and the
  plugin-based experiment workflow.
- [Guides](guides/index.md) — how the CLI, registries, plugins, and YAML
  configurations fit together.
- [API reference](api/index.md) — generated reference for the initial public
  core API.
- [Contributing](contributing.md) — local checks and pull-request guidance.

More involved runnable examples are maintained in the repository:
[VA estimation](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation),
[ORAGEN](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen), and
[affective states recognition](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition).

Source code and issue tracking are available on
[GitHub](https://github.com/markitantov/chimera_ml).
