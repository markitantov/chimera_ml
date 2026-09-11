# VA estimation

## Goal

Запустить audio или multimodal valence/arousal plugin из репозитория.

## What you will learn

Пример показывает регистрацию datamodules, models, losses, metrics, callbacks,
optimizer, scheduler и custom collates, после чего они используются командами
train/eval.

## Prerequisites

~~~bash
python -m pip install chimera-ml
python -m pip install -e examples/va_estimation
~~~

Нужны VA datasets, extracted features, annotations и machine-specific paths.
Некоторые configs содержат Telegram callback; удалите его или задайте
TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID.

## Project structure и configuration

Entry point — va_estimation = chimera_plugin:register. В configs находятся
audio_train/test и multimodal_train/test. Обновите data.params и annotation/
output paths.

~~~bash
chimera-ml validate-config -c examples/va_estimation/configs/multimodal_train.yaml
~~~

## Run

~~~bash
chimera-ml train -c examples/va_estimation/configs/multimodal_train.yaml
chimera-ml eval -c examples/va_estimation/configs/multimodal_test.yaml --checkpoint-path path/to/checkpoint.pt
~~~

Для audio-only используйте audio_* configs. Подробности — в
[example source и README](https://github.com/markitantov/chimera_ml/tree/main/examples/va_estimation).
