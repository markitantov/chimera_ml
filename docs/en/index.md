# Chimera ML

Chimera ML is a configuration-driven Python framework for building, training,
evaluating, and running uni-modal or multi-modal PyTorch experiments. The
runtime provides the trainer, registries, callbacks, loggers, and inference
pipeline; task-specific components can live in an installed plugin package.

## Start here

1. [Getting started](getting-started/index.md) — install Chimera ML and
   validate your first experiment configuration.
2. [Concepts](concepts/index.md) — understand configuration, registries,
   BuildContext, and plugins.
3. [User guide](user-guide/index.md) — configure training, evaluation, logging,
   inference, sweeps, and CLI workflows.
4. [Tutorials](tutorials/index.md) — use the real VA, ORAGEN, and affective
   states example projects.
5. [API reference](api/index.md) and [CLI reference](cli/reference.md) — find
   precise interfaces and command options.
6. [Developer guide](development/index.md) — contribute, test, document, and
   release the project.
7. [Releases](releases/index.md) — read changelog and migration notes.

## Core model boundary

A dataloader yields Batch, a model accepts Batch and returns ModelOutput, and
losses/metrics consume those objects. Experiments are described in YAML and
components are selected by registry key.

## Repository examples

The examples are real plugin packages and require their own data or model
artifacts:

- [VA estimation](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation)
- [ORAGEN](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen)
- [Affective states recognition](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition)

Install the published package with:

~~~bash
python -m pip install chimera-ml
~~~

The current package supports Python >=3.12,<3.13. Continue with
[Installation](getting-started/installation.md).
