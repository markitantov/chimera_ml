# VA estimation

## Задача и plugin

Plugin оценивает непрерывные valence и arousal. Доступны audio и multimodal training/evaluation: audio_train.yaml/audio_test.yaml и multimodal_train.yaml/multimodal_test.yaml. Готовой inference YAML нет. fusion_model, audio_*, fusion_* и va_* — plugin keys, а не TFS/RAAV/TOMAR.

## Установка и данные

~~~bash
python -m pip install -e .
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type models
chimera-ml registry list --type losses
~~~

Entry point: va_estimation = chimera_plugin:register, group chimera_ml.plugins. Audio dataset читает waveform при sr 16000, режет окна win_max_length/win_shift/win_min_length, использует VAD и создаёт cache в features_root. Labels должны содержать audio path и VA targets.

Multimodal DataModule использует modalities с feature path и emb_size: face 256, qwen_mm 256, qwen_v 128; audio задаётся отдельно. mask_audio позволяет учитывать отсутствие audio. Framewise callback читает callbacks[].params.ann_root и ann_split_dir — это annotation paths, не feature paths. Внешних corpus, challenge annotations и feature extraction command нет.

Перед запуском измените data.params.audio.path, data.params.modalities.*.path и emb_size, callback annotation paths, logging/callback log_path. Telegram callback требует TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID или должен быть удалён.

## Training и evaluation

~~~bash
chimera-ml validate-config --config-path examples/va_estimation/configs/audio_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/audio_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/audio_test.yaml --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

~~~bash
chimera-ml validate-config --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

Checkpoint callback пишет last.pt и monitored .pt, loggers/snapshot/framewise добавляют настроенные artifacts. Dataset-dependent training не запускался.
