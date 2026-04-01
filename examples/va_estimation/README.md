# VA Estimation Example Plugin

This directory contains an example plugin package for `chimera-ml`:

- audio VA pipeline (`audio_*` configs),
- multimodal fusion VA pipeline (`multimodal_*` configs),
- registry components loaded via entry point `chimera_ml.plugins`.

Code lives in `src/` (top-level modules `audio`, `fusion`, `metrics`, plus `chimera_plugin.py`), configs in `configs/`.

Packaging note: `setuptools` discovers all packages under `src` automatically (without explicit `include` filter).

## 1) Install

From repository root:

```bash
pip install -e examples/va_estimation
```

Alternative with Poetry:

```bash
poetry add ./examples/va_estimation
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

Before running, edit paths in YAML configs to match your machine:

- `data.params.*` paths (`dataset_path`, modality paths, CSV paths),
- callback annotation paths (`ann_root`, `ann_split_dir`),
- optional output pickle/submission paths in test configs.

Some train configs include `telegram_notifier_callback` and expect:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

If you do not need Telegram notifications, remove that callback from config.

## 4) Validate Config

```bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
```

## 5) Run Experiments

Multimodal training:

```bash
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Multimodal evaluation:

```bash
chimera-ml eval --config-path examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
```

Audio training:

```bash
chimera-ml train --config-path examples/va_estimation/configs/audio_train.yaml
```

Audio evaluation:

```bash
chimera-ml eval --config-path examples/va_estimation/configs/audio_test.yaml --checkpoint-path path/to/checkpoint.pt
```

Optional diagnostics:

```bash
chimera-ml doctor
```
