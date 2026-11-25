# Affective states recognition

## Задача и подготовка

Plugin распознаёт семь emotions и три sentiments по audio, video и text. Ключ
модели — lefsa_model; его prediction tensor содержит сначала emotion logits,
затем sentiment logits. Inference применяет softmax отдельно и создаёт
predictions artifact, описанный ниже.

~~~bash
python -m pip install -e .
python -m pip install -e examples/affective_states_recognition
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type inference_steps
chimera-ml validate-config --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
~~~

Training требует подготовленных CMU-MOSEI, MELD или RAMAS, label CSV, VAD
pickle, transcript/ASR columns и feature cache. Пути shipped YAML являются
placeholders: замените features_root, corpus roots, label/VAD paths и подготовьте
совместимые features. Mapping признаков: a wav2vec2, v emoaffectnet,
t jinaai. Окна имеют максимум 3 s, shift 2 s, minimum 2 s; video — 10 fps,
text max length 48. Проект не скачивает и не подготавливает datasets.

~~~bash
chimera-ml train --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/affective_states_recognition/configs/multimodal_test.yaml   --checkpoint-path "logs/your_experiment/your_run/checkpoints/last.pt"
~~~

## Inference DAG

Shipped inference.yaml использует parallel DAG и следующие зависимости:

~~~text
extract_audio_step
  -> vad_step -> build_segments_step -> transcribe_segments_step
                                      resolve_checkpoints_step ----------------> face_detection_step
                                           -> extract_features_step
                                           -> fusion_step
                                           -> write_json_predictions_step
                                           -> print_json_predictions_step
~~~

В конфигурации face detection ждёт resolve_checkpoints_step и
transcribe_segments_step; feature extraction ждёт face detection; fusion ждёт
features; writer ждёт fusion; print ждёт writer. Реальные этапы: audio decode,
Silero VAD, построение segments, Whisper transcription, YOLO face detection,
audio/video/text feature extraction, checkpoint-backed fusion и JSON output.

~~~bash
chimera-ml inference   --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4   --output prediction.json   --config-path examples/affective_states_recognition/configs/inference.yaml   --work-dir runs/affective
~~~

emoaffectnet и lefsa кэшируются в model_cache; Whisper и YOLO требуют
external weights. CPU поддерживается, но может быть медленным.

## Prediction artifact

Fusion записывает в InferenceContext.predictions объект:

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

Каждый segment содержит timing, duration, speech duration, weight для
aggregation, transcription, flags availability modalities, number of windows и
segment-level emotion/sentiment. Реализация строит weight как
max(speech_duration, segment duration) в seconds; поскольку speech
intersection ограничен segment, обычно это duration segment. Feature extraction
усредняет audio/video windows внутри segment и использует zero feature tensor,
если modality отсутствует. Финальные probabilities — weighted average
segment probabilities; это softmax probabilities, не calibrated confidence.

JSON writer добавляет outer envelope:

~~~json
{"input": "path/to/input.mp4", "predictions": {"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}}
~~~

При отсутствии features predictions artifact равен:

~~~json
{"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}
~~~

Следовательно, num_windows находится внутри каждого segment, а не в top-level
predictions. Training output и inference output — разные workflows.

## XAI и ограничения

FusionStep сам выполняет model call и не предоставляет on_prediction,
необходимый xai_step. Поэтому XAI для этого example не включён. Для отдельного
совместимого шага используйте общие руководства XAI и custom inference steps.
Внешние data, model weights, media decoding, network и writable cache остаются
обязательными prerequisites.
