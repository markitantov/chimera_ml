# Emotion Recognition Example Plugin

Example plugin for `chimera-ml` for multi-modal emotion and sentiment
recognition from audio, video, and text.

Targets:

- emotions: `neutral`, `happy`, `sad`, `anger`, `surprise`, `disgust`, `fear`
- sentiment: `negative`, `neutral`, `positive`
- corpora: `CMU-MOSEI`, `MELD`, `RAMAS`

Main files:

- training configs: `configs/multimodal_train.yaml`, `configs/multimodal_test.yaml`
- inference config: `configs/inference.yaml`
- code: `src/`

## 1) Install

```bash
pip install -e examples/emotion_recognition
```

## 2) Verify Plugin Registration

```bash
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

## 3) Update Config Paths

Before running, update dataset paths in:

- `configs/multimodal_train.yaml`
- `configs/multimodal_test.yaml`

Check:

- `data.params.features_root`
- `data.params.corpora.*.audio_root`
- `data.params.corpora.*.video_root`
- `data.params.corpora.*.labels_file_path`
- `data.params.corpora.*.vad_path`

## 4) Validate Config

```bash
chimera-ml validate-config --config-path examples/emotion_recognition/configs/multimodal_train.yaml
```

## 5) Run Experiments

```bash
chimera-ml train --config-path examples/emotion_recognition/configs/multimodal_train.yaml
```

```bash
chimera-ml eval \
  --config-path examples/emotion_recognition/configs/multimodal_test.yaml \
  --checkpoint-path path/to/checkpoint.pt
```

## 6) Inference

```bash
chimera-ml inference \
  --input video.mp4 \
  --output out.json \
  --config-path examples/emotion_recognition/configs/inference.yaml \
  --work-dir examples/emotion_recognition/123
```

The inference config resolves checkpoints into `model_cache/` and runs:

```yaml
- name: "resolve_checkpoints_step"
  params:
    cache_dir: "model_cache"
    checkpoints:
      emoaffectnet: "https://huggingface.co/markitantov/lefsa/resolve/main/emoaffectnet.pt"
      lefsa: "https://huggingface.co/markitantov/lefsa/resolve/main/lefsa.pt"
```

Pipeline: audio decode, VAD, transcription, face detection, feature extraction,
fusion, JSON output.

## 7) Related Publications

Markitantov M., Ryumina E., Kaya H., Karpov A. Multi-Modal Multi-Task
Affective States Recognition Based on Label Encoder Fusion //
In Proc. Interspeech 2025, pp. 3010-3014.
https://doi.org/10.21437/Interspeech.2025-2060

BibTeX:

```bibtex
@inproceedings{markitantov25_interspeech,
  title     = {{Multi-Modal Multi-Task Affective States Recognition Based on Label Encoder Fusion}},
  author    = {Maxim Markitantov and Elena Ryumina and Heysem Kaya and Alexey Karpov},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {3010--3014},
  doi       = {10.21437/Interspeech.2025-2060},
  issn      = {2958-1796}
}
```

Markitantov M., Ryumina E., Dvoynikova A., Karpov A. Multi-lingual approach
for multi-modal emotion and sentiment recognition based on triple fusion //
Information Fusion, 2026, vol. 132, article 104207.
https://doi.org/10.1016/j.inffus.2026.104207

BibTeX:

```bibtex
@article{markitantov2026triplefusion,
  title   = {Multi-lingual approach for multi-modal emotion and sentiment recognition based on triple fusion},
  author  = {Markitantov, Maxim and Ryumina, Elena and Dvoynikova, Anastasia and Karpov, Alexey},
  journal = {Information Fusion},
  volume  = {132},
  pages   = {104207},
  year    = {2026},
  doi     = {10.1016/j.inffus.2026.104207},
  url     = {https://doi.org/10.1016/j.inffus.2026.104207}
}
```
