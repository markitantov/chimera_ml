# ORAGEN

## Goal

Run the real ORAGEN example for audio-visual occlusion-robust gender
recognition and age estimation, including video inference.

## What you will learn

This example combines plugin models with a registry-driven inference pipeline,
remote/local checkpoint resolution, and optional DAG execution.

## Prerequisites

Install the plugin and its dependencies:

~~~bash
python -m pip install -e examples/oragen
~~~

Training requires ORAGEN-style cached features, labels, VAD files, corpora, and
the paths in the selected config. Inference additionally requires a readable
video, example dependencies, a torchcodec runtime or ffmpeg fallback, network
access on first checkpoint download (or local/cached weights), and writable
work space. Ultralytics resolves yolo26n.pt separately.

## Run inference

~~~bash
chimera-ml inference \
  --input examples/oragen/samples/f_25.mp4 \
  --output prediction.json \
  --config-path examples/oragen/configs/inference.yaml
~~~

The pipeline extracts mono 16 kHz audio, runs Silero VAD, samples video,
detects a face, builds fixed windows, resolves audio/image/fusion checkpoints,
extracts features, fuses predictions, and aggregates windows. The output step
writes JSON with input and predictions.

## Run training/evaluation

~~~bash
chimera-ml validate-config -c examples/oragen/configs/multimodal_train.yaml
chimera-ml train -c examples/oragen/configs/multimodal_train.yaml
chimera-ml eval -c examples/oragen/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

Audio configs use the same commands with audio_train.yaml and audio_test.yaml.

The authoritative paths, component keys, and model preparation steps remain in
the [ORAGEN example](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen).
