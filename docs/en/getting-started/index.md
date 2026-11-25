# Getting started

Follow these pages in order if Chimera ML is new to you:

1. [Installation](installation.md) — install the supported Python package and
   verify the CLI.
2. [Quick start](quickstart.md) — validate a shipped example configuration.
3. [First experiment](first-experiment.md) — understand the complete train and
   evaluation workflow.

After onboarding, read [Configuration](../concepts/configuration.md) and the
[User guide](../user-guide/index.md).


## Before you start

The core package supplies runtime infrastructure, not task data. Install a
plugin when your configuration refers to plugin datamodules, models, metrics,
callbacks, or inference steps. `validate-config` checks the framework shape; it
does not download a dataset, checkpoint, or optional plugin dependency.

For a first pass, use a repository example and follow its documented path
changes. Then replace its `data`, `model`, and task-specific component entries
with your own registered components.
