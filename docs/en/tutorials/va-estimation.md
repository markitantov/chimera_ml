# VA estimation

## Goal

Run the repository's audio or multimodal valence/arousal example plugin.

## What you will learn

You will see how a plugin registers datamodules, models, losses, metrics,
callbacks, an optimizer, a scheduler, and custom collates, then exposes them
through ordinary train/eval commands.

## Prerequisites

Install the package and the example:

~~~bash
python -m pip install chimera-ml
python -m pip install -e examples/va_estimation
~~~

The example requires its VA datasets, extracted features, annotations, and
machine-specific paths. Some configs also include Telegram notifications;
remove that callback or provide TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.

## Project structure and configuration

The package entry point is va_estimation = chimera_plugin:register. The
configs directory contains audio_train.yaml/audio_test.yaml and
multimodal_train.yaml/multimodal_test.yaml. Update data.params paths and any
annotation/output paths before running.

Each config selects plugin keys under data, model, loss, metrics, callbacks,
and optionally logging. Validate the exact config before starting:

~~~bash
chimera-ml validate-config -c examples/va_estimation/configs/multimodal_train.yaml
~~~

## Run

~~~bash
chimera-ml train -c examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval -c examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

Use the audio_* configs for the audio-only path.

## What happens and outputs

The CLI builds plugin components from registries and passes them through the
shared runtime. Configured callbacks can write checkpoints, a source snapshot,
predictions, and metric figures beneath the configured log path. Actual paths
and filenames come from the YAML.

Read the [example source and README](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation)
when changing dataset-specific preprocessing.
