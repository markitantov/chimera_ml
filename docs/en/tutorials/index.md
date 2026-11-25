# Examples

These guides describe the three example plugin projects shipped in examples/. They are concrete projects, not minimal toy applications. Install the relevant plugin, then prepare the external data and weights listed by its guide.

| Project | Task and inputs | Training | Evaluation | Inference | External requirements | Main concepts |
| --- | --- | --- | --- | --- | --- | --- |
| VA estimation | Valence/arousal from audio, or audio plus frame features | Audio and multimodal configs | Audio and multimodal configs | No shipped inference config | VA annotations, cached features, plugin checkpoints | Custom datasets, framewise callbacks, plugin loss/metrics |
| ORAGEN | Audio/video gender and age; optional mask labels in training | Audio and multimodal configs | Audio and multimodal configs | Shipped video pipeline | Corpora, VAD, cached .dat features for multimodal training; remote ORAGEN weights | Plugin models, checkpoint resolution, window aggregation |
| Affective states | Audio, video, and transcript features for emotion and sentiment | Multimodal config | Multimodal config | Shipped video pipeline | CMU-MOSEI/MELD/RAMAS data or prepared features; Whisper, YOLO, Hugging Face weights | Multi-task outputs, parallel inference DAG, feature caching |

Training and evaluation are not ready without external datasets and feature/annotation files. ORAGEN and affective-states provide sample media for inference, but inference still needs model weights and decoding/runtime dependencies.

For the framework-level first path, start with [First experiment](../getting-started/first-experiment.md).
