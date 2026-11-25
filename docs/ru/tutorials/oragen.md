# ORAGEN

## Задача и архитектура

ORAGEN демонстрирует определение пола и оценку возраста по аудио- и видеоданным с использованием мультимодального слияния, устойчивого к частичному перекрытию лица. Доступны аудио- и мультимодальное обучение и оценка, а также готовый видеоконвейер инференса. Используется модель плагина `agender_multimodal_model_v3`, а не модель слияния из ядра. Слияние возвращает логиты пола, за которыми следует один логит возраста. Инференс применяет softmax к полу и `sigmoid(age_logit) * 100` к возрасту, затем усредняет вероятности и возраст по окнам.

## Установка и проверка

Точка входа плагина — `oragen = chimera_plugin:register`. Установите плагин из корня репозитория и проверьте регистрацию командами ниже.

~~~bash
python -m pip install -e .
python -m pip install -e examples/oragen
chimera-ml plugins list
chimera-ml registry list --type inference_steps
chimera-ml registry list --type models
~~~

Важные каталоги и файлы: `configs/audio_{train,test}.yaml`, `configs/multimodal_{train,test}.yaml`, `configs/inference.yaml`, `scripts/`, а также `src/audio`, `src/fusion`, `src/common`, `src/inference`. Инференс требует `torchcodec` или резервного использования `ffmpeg`, Silero VAD, Ultralytics YOLO и весов Hugging Face; при первом запуске обычно нужен доступ к сети.

Аудиоконфигурации используют `features_root`, `corpus.data_root`, `labels_file_path` и `vad_path`; разметка содержит пол и возраст. Параметры окон: 16 kHz, 4 s, шаг 2 s. Мультимодальный режим использует VOXCELEB2 и BRAVE-MASKS, а также кэшированные записи с `acoustic_features` и `visual_features`. При отсутствии кэша нужны контрольные точки экстракторов признаков. Проект не скачивает корпуса автоматически.

## Данные, кэши и обучение

Измените `features_root`, пути `corpora.*`, `audio/image feature_extractor.checkpoint_path` и `train_subsets/val_subsets`. Команды:

~~~bash
chimera-ml validate-config --config-path examples/oragen/configs/audio_train.yaml
chimera-ml train --config-path examples/oragen/configs/audio_train.yaml
chimera-ml eval --config-path examples/oragen/configs/audio_test.yaml --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

~~~bash
chimera-ml validate-config --config-path examples/oragen/configs/multimodal_train.yaml
chimera-ml train --config-path examples/oragen/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/oragen/configs/multimodal_test.yaml --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

Ожидаемые артефакты — журналы, контрольные точки, сгруппированные метрики, необязательные графики матрицы ошибок и снимки в настроенном каталоге запуска.

## Видеоинференс

Конвейер из `inference.yaml` выполняется последовательно:

    extract_audio_step -> vad_step -> resolve_checkpoints_step
    -> face_detection_step -> build_windows_step -> extract_features_step
    -> fusion_step -> aggregate_windows_step -> print_json_predictions_step

~~~bash
chimera-ml inference --input examples/oragen/samples/f_25.mp4 \
  --output prediction.json --config-path examples/oragen/configs/inference.yaml \
  --work-dir runs/oragen
~~~

`resolve_checkpoints_step` кэширует веса аудио, изображения и слияния в `model_cache`; `YOLO yolo26n.pt` разрешается отдельно. CPU используется как резервный вариант. Нужны читаемый видеофайл, доступный для записи рабочий каталог, а также речь и лица, чтобы получить полезные окна. Итог содержит `gender`, `age`, `num_windows`, `windows` и `debug`; `gender` содержит `label`, `class_index` и `probabilities`, а каждое окно — `start_sec`/`end_sec`, поля пола, `age_pred`, `num_faces` и `num_frames`. При отсутствии окон `gender` и `age` равны `null`, а `status` — `no_predictions`.

## XAI и диагностика

`fusion_step` не предоставляет `on_prediction`, поэтому `xai_step` здесь не поддержан. Используйте [XAI](../user-guide/xai.md) и [пользовательские шаги инференса](../inference/custom-steps.md). Типичные проблемы связаны с плагином, кэшем и контрольной точкой, декодированием, некорректным видео, отсутствием лица или речи, устройством и путём вывода.


## Артефакт предсказаний и оболочка JSON

`AggregateWindowsStep` создаёт артефакт `predictions` с полями `gender`, `age` и
`num_windows`. При `include_windows` он также содержит `windows`, а при
`include_debug` — `debug`.

~~~json
{
  "gender": {
    "label": "female",
    "class_index": 0,
    "probabilities": {"female": 0.73, "male": 0.27}
  },
  "age": {"value": 34.2},
  "num_windows": 2,
  "windows": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 4.0,
      "gender_probs": {"female": 0.70, "male": 0.30},
      "gender_pred_index": 0,
      "gender_pred_label": "female",
      "age_pred": 33.8,
      "num_faces": 1,
      "num_frames": 4
    },
    {
      "index": 1,
      "start_sec": 4.0,
      "end_sec": 8.0,
      "gender_probs": {"female": 0.76, "male": 0.24},
      "gender_pred_index": 0,
      "gender_pred_label": "female",
      "age_pred": 34.6,
      "num_faces": 1,
      "num_frames": 4
    }
  ],
  "debug": {
    "speech_segments_sec": [{"start_sec": 0.0, "end_sec": 3.5}],
    "selected_face_track": null
  }
}
~~~

Слияние формирует запись окна для каждого элемента признаков. Вероятности
классов пола вычисляются функцией softmax; `age_pred` — сигмоида от логита
возраста, умноженная на `age_scale` 100. `AggregateWindowsStep` усредняет
вероятности и оценки возраста по окнам, затем выбирает класс пола с
максимальной средней вероятностью. Это не является калиброванной уверенностью.

Встроенное средство записи JSON оборачивает артефакт так:
{"input": "path/to/video.mp4", "predictions": {artifact}}. При отсутствии
окон `predictions` равен {"gender": null, "age": null, "num_windows": 0,
"windows": [], "status": "no_predictions"}.
