# CHIMERA-ML

CHIMERA-ML is a configuration-driven PyTorch framework for building, training, evaluating, and running uni-modal and multi-modal experiments. It supplies the runtime infrastructure—YAML construction, registries, Trainer, callbacks, loggers, checkpoints, and inference pipelines—while task-specific datasets and models may live in plugins.

## Start here

- [Getting started](getting-started/index.md): install the package, verify the environment, and run a first configuration check.
- [Training](user-guide/index.md): prepare data, configure an experiment, train, evaluate, and inspect artifacts.
- [Inference](user-guide/inference.md): execute a sequential or DAG pipeline over an input file.
- [Components](user-guide/models.md): inspect the Batch/ModelOutput contract, built-in fusion models, losses, metrics, callbacks, and loggers.
- [Examples](tutorials/index.md): reproduce the repository's VA, ORAGEN, and affective-states projects.
- [Extending](concepts/plugins.md): register your own datasets, models, losses, metrics, callbacks, or inference steps.
- [Reference](api/index.md): use the generated Python API and exact CLI/configuration references.

## The core idea

A training run follows `Dataset -> DataModule/DataLoader -> Batch -> Model -> ModelOutput -> loss/metrics -> Trainer -> checkpoints`. Inference is separate: `input -> InferencePipeline -> registered steps -> artifacts/predictions`. YAML selects components by registry key, and each component receives its `params` mapping.

The core package does not contain task datasets or silently invent their paths. Install the plugin required by the selected example or by your own configuration, update its resource paths, and run `validate-config` before training.

~~~bash
pip install chimera-ml
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list
~~~

See the [installation guide](getting-started/installation.md) for Poetry and development installs.
