# Affective states recognition

## Goal

Запустить multimodal emotion и sentiment example на audio, video и text.

## What you will learn

Пример показывает three-modality datamodule/model, multi-task loss/metrics,
inference DAG и remote checkpoint cache.

## Prerequisites

~~~bash
python -m pip install -e examples/affective_states_recognition
~~~

Configs требуют CMU-MOSEI, MELD или RAMAS paths, labels, VAD, features и
external model dependencies. Training не является out-of-the-box download.

## Run

~~~bash
chimera-ml validate-config -c examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml train -c examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml eval -c examples/affective_states_recognition/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

Для video inference:

~~~bash
chimera-ml inference \
  --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4 \
  --output prediction.json \
  --config-path examples/affective_states_recognition/configs/inference.yaml \
  --work-dir runs/inference
~~~

Pipeline декодирует audio, применяет VAD/segments/transcription/face detection,
features, fusion и пишет/печатает JSON. См. [example source и README](https://github.com/markitantov/chimera_ml/tree/main/examples/affective_states_recognition).
