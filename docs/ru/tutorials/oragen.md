# ORAGEN

ORAGEN демонстрирует audio-visual gender recognition и age estimation с occlusion-robust fusion. Есть audio и multimodal training/evaluation и готовый video inference. Используется plugin-модель agender_multimodal_model_v3, не core fusion model.

~~~bash
python -m pip install -e .
python -m pip install -e examples/oragen
chimera-ml plugins list
chimera-ml registry list --type inference_steps
chimera-ml registry list --type models
~~~

Важные каталоги: configs/audio_{train,test}.yaml, configs/multimodal_{train,test}.yaml, configs/inference.yaml, scripts/ и src/audio, src/fusion, src/common, src/inference. Inference требует torchcodec или ffmpeg fallback, Silero VAD, Ultralytics YOLO и Hugging Face weights; первый запуск обычно требует network.

Audio configs используют features_root, corpus data_root, labels_file_path, vad_path; labels дают gender и age. Окна: 16 kHz, 4 s, shift 2 s. Multimodal использует VOXCELEB2 и BRAVE-MASKS, cached records с acoustic_features и visual_features. При отсутствии cache нужны feature extractor checkpoints. Corpus проект не скачивает.

Измените features_root, corpora.* paths, audio/image feature_extractor.checkpoint_path и train_subsets/val_subsets. Команды:

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

## Video inference

Pipeline из inference.yaml последовательный:

    extract_audio_step -> vad_step -> resolve_checkpoints_step
    -> face_detection_step -> build_windows_step -> extract_features_step
    -> fusion_step -> aggregate_windows_step -> print_json_predictions_step

~~~bash
chimera-ml inference --input examples/oragen/samples/f_25.mp4 \
  --output prediction.json --config-path examples/oragen/configs/inference.yaml \
  --work-dir runs/oragen
~~~

resolve_checkpoints_step кэширует audio/image/fusion weights в model_cache; YOLO yolo26n.pt разрешается отдельно. CPU — fallback. Нужны readable video, writable work-dir и speech/face для полезных windows. Итог содержит gender, age, num_windows, windows, debug; gender содержит label, class_index, probabilities, окно — start_sec/end_sec, gender fields, age_pred, num_faces, num_frames. При отсутствии окон gender/age null и status no_predictions.

## XAI и troubleshooting

fusion_step не имеет on_prediction, поэтому xai_step здесь не поддержан. Используйте [XAI](../user-guide/xai.md) и [custom inference steps](../inference/custom-steps.md). Типовые ошибки: plugin/cache/checkpoint, decoding, invalid video, no face/speech, device и output path.


## Prediction artifact и JSON envelope

AggregateWindowsStep создаёт predictions artifact с gender, age и
num_windows. При include_windows он также содержит windows, при include_debug
— debug.

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

Fusion создаёт window record для каждого feature item. Gender probabilities —
softmax probabilities; age_pred — sigmoid(age logit), умноженный на age_scale
100. Aggregate усредняет window probabilities и age, затем выбирает gender с
максимальной средней probability. Это не calibrated confidence.

Built-in JSON writer оборачивает artifact как
{"input": "path/to/video.mp4", "predictions": {artifact}}. При отсутствии
windows predictions равен {"gender": null, "age": null, "num_windows": 0,
"windows": [], "status": "no_predictions"}.
