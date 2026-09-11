# ORAGEN

## Goal

Запустить реальный ORAGEN example для audio-visual gender recognition и age
estimation, включая video inference.

## What you will learn

Пример объединяет plugin models, registry-driven inference, local/remote
checkpoint resolution и optional DAG execution.

## Prerequisites

~~~bash
python -m pip install -e examples/oragen
~~~

Для training нужны cached features ORAGEN, labels, VAD, corpora и paths из
config. Для inference нужны readable video, dependencies примера, torchcodec
или ffmpeg fallback, сеть при первом скачивании checkpoint либо local/cache
weights и writable work space. Ultralytics отдельно разрешает yolo26n.pt.

## Run inference

~~~bash
chimera-ml inference \
  --input examples/oragen/samples/f_25.mp4 \
  --output prediction.json \
  --config-path examples/oragen/configs/inference.yaml
~~~

Pipeline декодирует mono 16 kHz audio, запускает VAD, sampling video, face
detection, windows, resolution checkpoint, feature extraction, fusion и
aggregation. Output step пишет JSON с input и predictions.

## Run training/evaluation

~~~bash
chimera-ml validate-config -c examples/oragen/configs/multimodal_train.yaml
chimera-ml train -c examples/oragen/configs/multimodal_train.yaml
chimera-ml eval -c examples/oragen/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

См. [ORAGEN example](https://github.com/markitantov/chimera_ml/tree/main/examples/oragen).
