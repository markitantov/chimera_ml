# VA estimation

## Task and plugin

This example estimates continuous valence and arousal (VA). It contains audio and multimodal training/evaluation families: audio_train.yaml/audio_test.yaml use the audio DataModule and window caches; multimodal_train.yaml/multimodal_test.yaml use the fusion DataModule and cached audio/image/frame features.

There is no shipped inference YAML. fusion_model, audio_*, fusion_*, and va_* are plugin registrations, not the core TFS/RAAV/TOMAR architectures.

## Install and inspect registration

From the repository root:

~~~bash
python -m pip install -e .
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type models
chimera-ml registry list --type losses
~~~

The entry point is va_estimation = chimera_plugin:register in the chimera_ml.plugins group.

## Project and data

    examples/va_estimation/
    ├── configs/audio_{train,test}.yaml
    ├── configs/multimodal_{train,test}.yaml
    └── src/
        ├── chimera_plugin.py
        ├── audio/data, audio/models, audio/loss
        ├── fusion/data, fusion/models, fusion/loss, fusion/callbacks
        └── metrics/

The datasets read configured waveform/feature paths; they do not download a corpus. Audio is read at sr 16000, split with win_max_length, win_shift and win_min_length, and filtered with VAD metadata when present. Labels must provide the audio path and VA targets. A feature index and window cache are created under features_root.

The multimodal DataModule reads the modalities mapping. The shipped adapted configuration uses face, qwen_mm, and qwen_v with emb_size 256, 256, and 128. Audio has its own path and emb_size. mask_audio: true allows missing audio to be represented by the configured mask. Framewise evaluation reads annotations from callbacks[].params.ann_root and ann_split_dir; these are not feature paths.

Change these placeholders before use:

| Field | Meaning |
| --- | --- |
| data.params.audio.path | Audio feature/window cache |
| data.params.modalities.*.path | Per-modality frame features |
| data.params.*.emb_size | Actual feature width |
| callbacks[].params.ann_root | VA challenge annotation root |
| callbacks[].params.ann_split_dir | Annotation split |
| logging[].params.log_path and callback log_path | Logs/checkpoints |
| telegram_notifier_callback | Optional Telegram environment credentials |

The example contains no external VA corpus, challenge annotations, or feature download command. Treat repository-relative paths as placeholders.

## Validate, train, and evaluate

~~~bash
chimera-ml validate-config --config-path examples/va_estimation/configs/audio_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/audio_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/audio_test.yaml \
  --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

~~~bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/multimodal_test.yaml \
  --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

The checkpoint callback writes last.pt and monitored .pt files below the run directory. Loggers, snapshots, and the framewise callback add configured artifacts. Evaluation needs a state dict compatible with the selected audio or fusion model. Remove the Telegram callback or export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Dataset-dependent training was not executed.
