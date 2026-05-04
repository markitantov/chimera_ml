# ORAGEN Example Plugin

This directory contains an example plugin package for `chimera-ml` based on
the ORAGEN approach from the original project:

- project page: https://smil-spcras.github.io/ORAGEN/
- original repository: https://github.com/markitantov/ORAGEN

ORAGEN is an audio-visual approach for occlusion-robust gender recognition and
age estimation. In the original work, the bimodal model uses intermediate
features from unimodal transformer-based models together with Multi-Task
Cross-Modal Attention (MTCMA) blocks and can jointly predict gender, age, and
protective mask type.

This `chimera-ml` example provides:

- audio pipeline (`audio_*` configs),
- multimodal fusion pipeline (`multimodal_*` configs),
- registry-driven video inference pipeline (`inference.yaml`),
- registry components loaded via entry point `chimera_ml.plugins`.

Code lives in `src/`, training configs in `configs/`, the inference config in
`configs/inference.yaml`, and the helper export script in `scripts/`.

Registered components:

- datamodules: `agender_audio_datamodule`, `agender_fusion_datamodule`
- models: `agender_audio_w2v2_model`, `agender_audio_hubert_model`,
  `agender_sota_model`, `agender_image_vit_gsa_model`,
  `agender_image_vit_dpal_model`, `agender_multimodal_model_v1`,
  `agender_multimodal_model_v2`, `agender_multimodal_model_v3`
- loss: `agender_loss`
- metrics: `age_mae_metric`, `age_pcc_metric`,
  `gender_prf_macro_metric`, `mask_uar_metric`
- callback: `grouping_callback`
- inference steps: `extract_audio_step`, `vad_step`, `face_detection_step`,
  `build_windows_step`, `extract_features_step`, `fusion_step`,
  `aggregate_windows_step`
- built-in inference output steps used by the config:
  `print_json_predictions_step`, `write_json_predictions_step`

Packaging note: `setuptools` discovers all packages under `src`
automatically.

## 1) Install

From repository root:

```bash
pip install -e examples/oragen
```

Alternative with Poetry:

```bash
poetry add ./examples/oragen
```

After installation, `chimera-ml` discovers this plugin automatically.

## 2) Verify Plugin Registration

```bash
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type models
```

Note: `chimera_ml` command alias also works, but docs use `chimera-ml`.

## 3) Update Config Paths

Before running, edit paths in YAML configs to match your machine.

Audio configs (`configs/audio_train.yaml`, `configs/audio_test.yaml`):

- `data.params.features_root`
- `data.params.corpora.*.data_root`
- `data.params.corpora.*.labels_file_path`
- `data.params.corpora.*.vad_path`

Multimodal configs (`configs/multimodal_train.yaml`,
`configs/multimodal_test.yaml`):

- `data.params.features_root`
- `data.params.corpora.*.data_root`
- `data.params.corpora.*.labels_file_path`
- `data.params.corpora.*.vad_path`
- `data.params.audio_feature_extractor.checkpoint_path`
- `data.params.image_feature_extractor.checkpoint_path`

The multimodal pipeline expects ORAGEN-style cached `.dat` files with
`acoustic_features` and `visual_features`. Generate these features with the
original ORAGEN extraction pipeline first, or point `features_root` to an
existing cache.

## 4) Validate Config

```bash
chimera-ml validate-config --config-path examples/oragen/configs/multimodal_train.yaml
```

## 5) Run Experiments

Inference from video:

```bash
chimera-ml inference \
  --input video.mp4 \
  --output out.json \
  --config-path examples/oragen/configs/inference.yaml
```

The ready-to-run inference config expects:

- a readable input video path
- `ffmpeg` available in the environment
- network access for the first checkpoint download, or already cached/local ORAGEN checkpoints
- enough writable space in the inference work directory for `model_cache/`

Current inference flow:

- convert input video to mono `16 kHz` audio
- run Silero VAD
- sample video at `1 fps`
- detect the best face on each sampled frame with `YOLO("yolo26n.pt")`
- build fixed `4s` windows
- resolve ORAGEN audio/image/fusion checkpoints with `resolve_checkpoints_step`
- extract audio and image features
- run `agender_multimodal_model_v3`
- aggregate window predictions for the whole video

The ORAGEN inference config resolves model weights into the shared `checkpoints`
artifact before downstream steps run:

```yaml
- name: "resolve_checkpoints_step"
  params:
    cache_dir: "model_cache"
    checkpoints:
      audio: "https://huggingface.co/markitantov/ORAGEN/resolve/main/audio_model.pt"
      image: "https://huggingface.co/markitantov/ORAGEN/resolve/main/image_model.pt"
      fusion: "https://huggingface.co/markitantov/ORAGEN/resolve/main/multimodal_model.pt"

- name: "extract_features_step"
  params:
    audio_checkpoint_key: "audio"
    image_checkpoint_key: "image"

- name: "fusion_step"
  params:
    checkpoint_key: "fusion"
```

On the first run, `resolve_checkpoints_step` prints where each checkpoint is
downloaded and stores it under `<work_dir>/model_cache/`. Later runs reuse the
resolved local file from the same cache directory.

If a checkpoint key is missing from `artifacts["checkpoints"]`, or the resolved
path is not a file, inference now fails early with a direct message from the
step that requested it.

`yolo26n.pt` is still resolved separately by Ultralytics. In this ORAGEN config,
that ends up under the same inference work-directory `model_cache/` area.

DAG inference:

You can also describe inference as a small DAG instead of a purely sequential
list of steps. For that, annotate steps with:

- each step still has the same shape as before: `name` plus optional `params`
- set `pipeline.parallel: true` to switch the whole inference config into DAG mode
- without `pipeline.parallel: true`, steps still run sequentially in config order
- `after`: which earlier steps must finish before this one starts in DAG mode

By default, a DAG node is identified by its step `name`. If the same step name
is reused multiple times in one DAG, assign explicit `id` values only to those
repeated steps and reference them from `after`.

If two DAG steps write the same artifact key, the pipeline raises an error.
If DAG mode is enabled but no step uses `after`, the builder emits a warning.

Example where only repeated steps need `id`:

```yaml
pipeline:
  parallel: true

steps:
  - name: extract_audio_step
    params:
      sample_rate: 16000

  - name: vad_step
    after: [extract_audio_step]

  - id: detector_fast
    name: face_detection_step
    after: [extract_audio_step]
    params:
      conf: 0.25
      model: yolo26n.pt
      fps: 1.0

  - id: detector_accurate
    name: face_detection_step
    after: [extract_audio_step]
    params:
      conf: 0.5
      model: yolo26n.pt
      fps: 1.0

  - name: build_windows_step
    after: [vad_step, detector_fast, detector_accurate]
    params:
      win_max_length: 4
      win_shift: 2
      win_min_length: 1
```

Multimodal training:

```bash
chimera-ml train --config-path examples/oragen/configs/multimodal_train.yaml
```

Multimodal evaluation:

```bash
chimera-ml eval --config-path examples/oragen/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
```

Audio training:

```bash
chimera-ml train --config-path examples/oragen/configs/audio_train.yaml
```

Audio evaluation:

```bash
chimera-ml eval --config-path examples/oragen/configs/audio_test.yaml --checkpoint-path path/to/checkpoint.pt
```

Optional diagnostics:

```bash
chimera-ml doctor
```

## 6) Related publications

Markitantov M., Ryumina E., Karpov A. Audio-visual occlusion-robust gender recognition and age estimation approach based on multi-task cross-modal attention. // Expert Systems with Applications. 2026. vol. 296. ID 127473. https://doi.org/10.1016/j.eswa.2025.127473

BibTeX:

```bibtex
@article{markitantov2026oragen,
  author = {Markitantov, Maxim and Ryumina, Elena and Karpov, Alexey},
  title = {Audio-visual occlusion-robust gender recognition and age estimation approach based on multi-task cross-modal attention},
  journal = {Expert Systems with Applications},
  volume = {296},
  pages = {127473},
  year = {2026},
  month = jan,
  doi = {10.1016/j.eswa.2025.127473},
  url = {https://doi.org/10.1016/j.eswa.2025.127473}
}
```
