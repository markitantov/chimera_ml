# Распознавание аффективных состояний

## Задача и проект



Плагин распознаёт семь эмоций и три тональных класса по аудио-, видео- и текстовым данным. Ключ
модели — `lefsa_model`; её тензор предсказаний содержит сначала логиты эмоций,
затем логиты тональности. Инференс отдельно применяет softmax и создаёт
описанный ниже артефакт `predictions`.

## Установка и подготовка данных

~~~bash
python -m pip install -e .
python -m pip install -e examples/affective_states_recognition
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type inference_steps
chimera-ml validate-config --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
~~~

Для обучения нужны подготовленные CMU-MOSEI, MELD или RAMAS, CSV с разметкой,
pickle-файл VAD, столбцы транскрипции/ASR и кэш признаков. Пути в поставляемом
YAML являются заполнителями: замените `features_root`, корневые пути корпусов и
пути к разметке/VAD, а также подготовьте совместимые признаки. Отображение
признаков: `a` — wav2vec2, `v` — emoaffectnet, `t` — jinaai. Максимальная
длина окна — 3 s, шаг — 2 s, минимальная длина — 2 s; видео обрабатывается с
частотой 10 fps, максимальная длина текста — 48. Проект не скачивает и не
подготавливает наборы данных.

~~~bash
chimera-ml train --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/affective_states_recognition/configs/multimodal_test.yaml   --checkpoint-path "logs/your_experiment/your_run/checkpoints/last.pt"
~~~

## DAG инференса

Поставляемый `inference.yaml` использует параллельный DAG со следующими зависимостями:

~~~text
extract_audio_step
        |
        v
vad_step
        |
        v
build_segments_step
        |
        v
transcribe_segments_step    resolve_checkpoints_step
        |                            |
        +------------+---------------+
                     |
                     v
             face_detection_step
                     |
                     v
             extract_features_step
                     |
                     v
                 fusion_step
                     |
                     v
         write_json_predictions_step
                     |
                     v
         print_json_predictions_step
~~~

Извлечение аудио и разрешение контрольных точек — независимые корневые шаги.
В конфигурации обнаружение лиц зависит одновременно от
`resolve_checkpoints_step` и `transcribe_segments_step`. Последующие шаги
следуют объявленным зависимостям: извлечение признаков зависит от обнаружения
лиц; слияние — от признаков; средство записи — от слияния; печать — от средства
записи. Фактические этапы: декодирование аудио, Silero VAD, построение
сегментов, транскрибация Whisper, обнаружение лиц YOLO, извлечение аудио-,
видео- и текстовых признаков, слияние с использованием контрольной точки и
вывод JSON.

~~~bash
chimera-ml inference   --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4   --output prediction.json   --config-path examples/affective_states_recognition/configs/inference.yaml   --work-dir runs/affective
~~~

Настроенный конвейер кэширует `emoaffectnet` и `lefsa` в `model_cache`; Whisper и YOLO требуют
внешние веса. CPU поддерживается, но выполнение может быть медленным.

## Артефакт предсказаний

Слияние записывает в `InferenceContext.predictions` следующий объект:

~~~json
{
  "emotion": {
    "label": "happy",
    "class_index": 1,
    "probabilities": {"neutral": 0.10, "happy": 0.60, "sad": 0.04, "anger": 0.03, "surprise": 0.09, "disgust": 0.06, "fear": 0.08}
  },
  "sentiment": {
    "label": "positive",
    "class_index": 2,
    "probabilities": {"negative": 0.08, "neutral": 0.17, "positive": 0.75}
  },
  "text": "example transcript",
  "segments": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 3.0,
      "duration_sec": 3.0,
      "speech_duration_sec": 2.4,
      "weight": 3.0,
      "text": "example transcript",
      "has_text": true,
      "has_speech": true,
      "has_faces": true,
      "num_windows": 1,
      "emotion": {"label": "happy", "class_index": 1, "probabilities": {"neutral": 0.10, "happy": 0.60, "sad": 0.04, "anger": 0.03, "surprise": 0.09, "disgust": 0.06, "fear": 0.08}},
      "sentiment": {"label": "positive", "class_index": 2, "probabilities": {"negative": 0.08, "neutral": 0.17, "positive": 0.75}}
    }
  ]
}
~~~

Каждый сегмент содержит временные границы, длительность, длительность речи,
вес для агрегации, транскрипцию, флаги наличия модальностей, число окон и
эмоцию/тональность на уровне сегмента. Реализация вычисляет `weight` как
максимум `speech_duration` и длительности сегмента в секундах; поскольку
пересечение речи ограничено сегментом, обычно это длительность сегмента.
Извлечение признаков усредняет аудио- и видеоокна внутри сегмента и использует
нулевой тензор признаков, если модальность отсутствует. Итоговые вероятности —
взвешенное среднее вероятностей сегментов; это вероятности softmax, а не
калиброванная уверенность.

Средство записи JSON добавляет внешнюю оболочку:

~~~json
{"input": "path/to/input.mp4", "predictions": {"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}}
~~~

При отсутствии признаков артефакт `predictions` имеет вид:

~~~json
{"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}
~~~

Следовательно, `num_windows` находится внутри каждого сегмента, а не на
верхнем уровне `predictions`. Результат обучения и результат инференса относятся
к разным сценариям работы.

## XAI и ограничения

`FusionStep` не предоставляет `on_prediction`, необходимый для `xai_step`, поэтому XAI для этого примера не включён по умолчанию. Для отдельного
совместимого шага используйте общее руководство по XAI и пользовательские шаги
инференса. Внешние данные и веса модели, декодирование медиа, сетевой доступ и
доступный для записи кэш остаются обязательными предварительными условиями.
