# Getting started

Use this sequence for a first project. The core package supplies the runtime and built-in registries; task-specific datasets and datamodules usually come from a plugin.

1. [Installation](installation.md): install Python 3.12 and CHIMERA-ML, then verify the environment.
2. [Quick start](quickstart.md): inspect CLI commands, registries, and a real plugin configuration without training.
3. [First experiment](first-experiment.md): install a plugin, prepare its data, validate, train, find artifacts, and evaluate a checkpoint.
4. [Essential concepts](essential-concepts.md): understand the objects exchanged by the training loop.

After that, use the [Training overview](../training/index.md) and its linked workflow pages.

## Training, evaluation, and inference

- **Training** calls `Trainer.fit`, updates model parameters, and writes checkpoints and logs. Training batches must contain targets.
- **Evaluation** calls `Trainer.evaluate` with `train=False`; it never performs optimizer updates. It loads a compatible checkpoint and can cache predictions or features.
- **Inference** is a separate `InferencePipeline` over an input file. It is not a substitute for checkpoint evaluation and does not require the training loop's loss or optimizer.

`validate-config` checks the framework-level YAML shape and required sections. It does not download datasets, resolve plugin dependencies, or prove that task-specific paths exist.
