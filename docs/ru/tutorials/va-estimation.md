# Оценка VA

## Задача и плагин

Плагин оценивает непрерывные значения валентности и активации. Доступны аудио- и мультимодальное обучение и оценка: `audio_train.yaml`/`audio_test.yaml` и `multimodal_train.yaml`/`multimodal_test.yaml`. Готовой конфигурации YAML для инференса нет. `fusion_model`, `audio_*`, `fusion_*` и `va_*` — ключи плагина, а не TFS/RAAV/TOMAR.

## Установка и проверка регистрации

~~~bash
python -m pip install -e .
python -m pip install -e examples/va_estimation
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type models
chimera-ml registry list --type losses
~~~

Точка входа: `va_estimation = chimera_plugin:register`, группа `chimera_ml.plugins`. Аудионабор данных читает форму сигнала с частотой `sr` 16000, нарезает окна `win_max_length`/`win_shift`/`win_min_length`, использует VAD и создаёт кэш в `features_root`. Разметка должна содержать путь к аудио и целевые значения VA.

Мультимодальный `DataModule` использует модальности с путями к признакам и `emb_size`: `face` 256, `qwen_mm` 256, `qwen_v` 128; аудио задаётся отдельно. `mask_audio` позволяет учитывать отсутствие аудио. Покадровый колбэк читает `callbacks[].params.ann_root` и `ann_split_dir`; это пути к аннотациям, а не к признакам. Внешние корпуса, аннотации соревнования и команда извлечения признаков не поставляются.

Перед запуском измените `data.params.audio.path`, `data.params.modalities.*.path` и `emb_size`, пути аннотаций колбэка, а также `log_path` журналировщика и колбэков. Колбэк Telegram требует `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`; иначе удалите его из конфигурации.

## Проект и данные

Проект содержит конфигурации `audio_{train,test}.yaml` и `multimodal_{train,test}.yaml`, а также исходные пакеты для аудио, слияния и метрик. Наборы данных читают настроенные пути к волновым формам и признакам и не скачивают корпуса.

| Поле | Назначение |
| --- | --- |
| `data.params.audio.path` | Кэш аудиопризнаков или окон |
| `data.params.modalities.*.path` | Кадровые признаки отдельной модальности |
| `data.params.*.emb_size` | Фактическая ширина признаков |
| `callbacks[].params.ann_root` | Корень аннотаций VA |
| `callbacks[].params.ann_split_dir` | Выборка аннотаций |
| `logging[].params.log_path` и `callback log_path` | Журналы и контрольные точки |
| `telegram_notifier_callback` | Необязательные учётные данные Telegram из переменных окружения |

Внешний корпус VA, аннотации соревнования и команда скачивания признаков в примере не поставляются; относительные пути репозитория являются заполнителями.

## Проверка, обучение и оценка

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

Колбэк контрольных точек записывает `last.pt` и контролируемые файлы `.pt`, а журналировщики, снимок и покадровый колбэк добавляют настроенные артефакты. Обучение, зависящее от внешнего набора данных, не запускалось.
