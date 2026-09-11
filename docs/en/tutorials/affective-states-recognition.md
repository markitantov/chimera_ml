# Affective states recognition

## Goal

Run the multimodal emotion and sentiment example over audio, video, and text.

## What you will learn

This example demonstrates a three-modality datamodule and model, multi-task
losses/metrics, an inference DAG, and remote checkpoint caching.

## Prerequisites

~~~bash
python -m pip install -e examples/affective_states_recognition
~~~

The configs require the CMU-MOSEI, MELD, or RAMAS paths, labels, VAD data,
features, and external model dependencies. They are not an out-of-the-box
training dataset download.

## Run

~~~bash
chimera-ml validate-config -c examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml train -c examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml eval -c examples/affective_states_recognition/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

For video inference:

~~~bash
chimera-ml inference \
  --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4 \
  --output prediction.json \
  --config-path examples/affective_states_recognition/configs/inference.yaml \
  --work-dir runs/inference
~~~

The pipeline decodes audio, applies VAD and segmenting, transcribes, detects
faces, extracts features, fuses emotion/sentiment predictions, then writes and
prints JSON. Its config explicitly declares output dependencies because it uses
parallel mode.

See the [example source and README](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition)
for corpus preparation and model paths.
