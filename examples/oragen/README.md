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
- registry components loaded via entry point `chimera_ml.plugins`.

Code lives in `src/`, configs in `configs/`, and the helper export script in
`scripts/`.

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
