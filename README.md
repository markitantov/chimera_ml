# Chimera ML

`Chimera ML (Cross-modal HIerarchical Merging of Embeddings and Representations)` is a lightweight training framework for configurable uni-modal and multi-modal experiments.
It is built around a small set of registries and a YAML-driven CLI, so project-specific code lives in an external plugin package while training, evaluation, logging, checkpointing, and callback orchestration stay inside the library.

This README reflects the current usage pattern of the library:

```bash
chimera_ml train --config-path path/to/cfg.yaml
chimera_ml eval --config-path path/to/cfg.yaml --checkpoint-path path/to/checkpoint.pt
```

## What the library does

`Chimera ML` provides:

- registry-based construction of datamodules, models, losses, metrics, optimizers, schedulers, callbacks, collates, and loggers;
- YAML-driven experiment setup;
- a PyTorch training loop with mixed precision, gradient clipping, schedulers, and callback hooks;
- evaluation over `train` / `val` / `test` loaders;
- checkpointing and snapshotting;
- console and MLflow logging;
- support for multimodal batches and modality-aware collation;
- plugin loading through Python entry points, so external projects can register their own components without modifying the library itself.

The intended workflow is:

1. `Chimera ML` is installed as a reusable library.
2. A task-specific project defines and registers its own datamodules, models, losses, callbacks, etc.
3. YAML configs reference those components by name.
4. The CLI reads the config, builds everything from registries, and runs training or evaluation.

---

## Installation

### 1. Install the library itself

Inside the `chimera_ml` repository:

```bash
poetry install
```

This installs the package and exposes the CLI entry point:

```bash
chimera_ml
```

### 2. Install it in a project that uses custom components

A typical project depends on `Chimera ML` as a local package:

```toml
[tool.poetry.dependencies]
chimera-ml = { path = "../chimera_ml", develop = true }
```

Then install project dependencies:

```bash
poetry install
```

That is the setup used in the ABAW project as well: the task repository contains task-specific code, while `Chimera ML` provides the training runtime.

---

## High-level architecture

The framework is built around registries defined in `chimera_ml.core.registry`:

- `DATAMODULES`
- `MODELS`
- `LOSSES`
- `METRICS`
- `OPTIMIZERS`
- `SCHEDULERS`
- `CALLBACKS`
- `COLLATES`
- `LOGGERS`

Each registry maps a string name from YAML to a factory function or class constructor.

At runtime the CLI does the following:

1. loads the YAML config;
2. sets the random seed;
3. creates the experiment / run name;
4. builds the datamodule from `data.name` and `data.params`;
5. optionally asks the datamodule for model context via `get_model_context()`;
6. builds the model from `model.name` and `model.params`;
7. builds loss, metrics, optimizer, scheduler, loggers, and callbacks;
8. runs training or evaluation using `Trainer`.

This means the library itself does not need to know anything about a specific task such as emotion recognition, audiovisual fusion, or sequence-to-sequence regression. All task-specific behavior is provided by registered components.

---

## How component registration works

### Built-in components

Some components are shipped with `chimera_ml` itself, for example:

- optimizers from `chimera_ml.training.optimizers`
- schedulers from `chimera_ml.training.schedulers`
- callbacks like `checkpoint_callback`, `early_stopping_callback`, `snapshot_callback`
- built-in fusion models
- basic regression / classification losses and metrics

These are registered automatically when `chimera_ml` is imported.

### Project-specific components

External projects register their own components using the registry decorators.

Example: registering a datamodule

```python
from chimera_ml.core.registry import DATAMODULES

@DATAMODULES.register("my_custom_datamodule")
def my_custom_datamodule(**params):
    return MyCustomDatamodule(**params)
```

Example: registering a model

```python
from chimera_ml.core.registry import MODELS

@MODELS.register("my_model")
def my_model(**params):
    return MyModel(**params)
```

Example: registering a custom callback

```python
from chimera_ml.core.registry import CALLBACKS

@CALLBACKS.register("my_custom_callback")
def my_custom_callback(**params):
    return MyCustomCallback(**params)
```

After that, YAML can reference these names directly.

---

## How plugin loading works

For external projects, the recommended mechanism is Python entry points.

In the task-specific project's `pyproject.toml`:

```toml
[tool.poetry.plugins."chimera_ml.plugins"]
my_project = "chimera_plugin:register"
```

Then the project provides a function like:

```python
# chimera_plugin.py

def register():
    import my_project.datamodules
    import my_project.models
    import my_project.losses
    import my_project.metrics
    import my_project.callbacks
```

The goal of that function is simply to import modules that execute registration decorators.

When `Chimera ML` starts, it loads all entry points from the `chimera_ml.plugins` group. If an entry point resolves to a callable, it is executed automatically.

This is why commands such as:

```bash
chimera_ml train --config-path path/to/cfg.yaml
```

can construct components that do not exist inside the `chimera_ml` repository itself, as long as the project plugin is installed and registered.

---

## YAML config structure

The framework expects a config with sections similar to the following:

```yaml
seed: 0

experiment_info:
  params:
    experiment_name: "10th_ABAW"
    run_name: "wavlm"
    datetime_format: "%Y-%m-%d_%H-%M"

data:
  name: "my_custom_datamodule"
  params:
    ...

model:
  name: "my_module"
  params:
    ...

train:
  params:
    epochs: 50
    ...

optimizer:
  name: "adamw_optimizer"
  params:
    ...

loss:
  name: "mse_loss"
  params:
    ...

metrics:
  - name: "va_ccc_metric"
    params: 
    ...

callbacks:
  - name: "checkpoint_callback"
    params:
      log_path: "logs"
      monitor: "val_framewise/va_ccc_metric"
      mode: "max"
  - name: "my_custom_callback"
    params:
      ...

logging:
  - name: "console_file_logger"
    params:
      log_path: "logs"
  - name: "mlflow_logger"
    params:
      tracking_uri: "mlruns"
```

### Section semantics

- `data`: builds a datamodule from the `DATAMODULES` registry.
- `model`: builds a model from the `MODELS` registry.
- `optimizer`: builds an optimizer and injects the current model.
- `scheduler`: optional, receives the already built optimizer.
- `loss`: builds a loss object.
- `metrics`: list of metric configs.
- `callbacks`: list of callback configs.
- `logging`: list-based section; can contain multiple loggers.
- `train.params`: converted into `TrainConfig`.

### Important note about names

Registry keys are lowercased by the builder, so YAML names should be treated as case-insensitive, but in practice it is best to keep them lowercase and consistent.

---

## CLI usage

### Training

Use:

```bash
chimera_ml train --config-path path/to/cfg.yaml
```

This command:

- loads the training config;
- creates the datamodule and model;
- builds optimizer, scheduler, loss, metrics, callbacks, and loggers;
- runs `Trainer.fit(...)`;
- saves checkpoints through `checkpoint_callback` if configured.

### Evaluation

Use:

```bash
chimera_ml eval --config-path path/to/cfg.yaml --checkpoint-path path/to/checkpoint.pt
```

This command:

- loads the config;
- builds the same datamodule and model;
- restores model weights from `checkpoint_path`;
- runs evaluation on every available split returned by the datamodule;
- executes callbacks in eval mode as well.

This matches the intended usage in downstream projects, for example:

```bash
chimera_ml train --config-path path/to/cfg.yaml
```

and

```bash
chimera_ml eval \
  --config-path path/to/cfg.yaml \
  --checkpoint-path path/to/checkpoint.pt
```

### Optional CLI flags

`train` supports:

- `--config-path`
- `--class-names`

`eval` supports:

- `--config-path`
- `--checkpoint-path`
- `--with-features`
- `--class-names`

`--with-features` is useful if your model exposes features in `ModelOutput.aux["features"]` or if a callback expects extracted representations.