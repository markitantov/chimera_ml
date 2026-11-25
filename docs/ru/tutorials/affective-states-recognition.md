# Affective states recognition

Plugin распознаёт семь emotions и три sentiments по audio, video и text. Модель lefsa_model возвращает preds с emotion logits, затем sentiment logits; inference отдельно применяет softmax и усредняет probabilities окон.

    examples/affective_states_recognition/
    ├── configs/{multimodal_train,multimodal_test,inference}.yaml
    ├── samples/
    └── src/{chimera_plugin.py,fusion,feature_extractors,inference,common}

Установка:

    python -m pip install -e .
    python -m pip install -e examples/affective_states_recognition
    chimera-ml plugins list
    chimera-ml registry list --type datamodules
    chimera-ml registry list --type inference_steps
    chimera-ml validate-config --config-path examples/affective_states_recognition/configs/multimodal_train.yaml

Training требует CMU-MOSEI, MELD или RAMAS, labels CSV, VAD pickle, transcript/ASR и feature cache. Пути в YAML — placeholders: data/CMUMOSEI/vocals, data/CMUMOSEI/Video_chunk/{subset}, data/MELD/vocals/{subset}, data/MELD/videos/{subset}, data/RAMAS/vocals, data/RAMAS/videos и соответствующие labels/VAD paths. Feature mapping: a wav2vec2, v emoaffectnet, t jinaai; окна 3 s, shift 2 s, minimum 2 s, video 10 fps, text max 48. Download/preprocessing command нет.

    chimera-ml train --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
    chimera-ml eval --config-path examples/affective_states_recognition/configs/multimodal_test.yaml --checkpoint-path logs/<experiment>/<run>/checkpoints/last.pt

## Video inference

Config — parallel DAG; after задаёт зависимости producers/consumers:

    extract_audio_step -> vad_step -> build_segments_step -> transcribe_segments_step
    resolve_checkpoints_step -> face_detection_step -> extract_features_step
    extract_features_step -> fusion_step -> write_json_predictions_step
    write_json_predictions_step -> print_json_predictions_step

Этапы: audio decode, Silero VAD, segmentation, Whisper transcription, YOLO face detection, audio/video/text feature extraction, checkpoint fusion, JSON. emoaffectnet и lefsa кэшируются; Whisper и YOLO требуют external weights.

    chimera-ml inference \\
      --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4 \\
      --output prediction.json \\
      --config-path examples/affective_states_recognition/configs/inference.yaml \\
      --work-dir runs/affective

Final JSON содержит emotion и sentiment (label, class_index, probabilities) и num_windows. Это softmax probabilities, не calibrated confidence; при отсутствии windows emotion/sentiment null.

## XAI

FusionStep не предоставляет on_prediction, поэтому xai_step для example не включён. Для совместимого custom step используйте [XAI](../user-guide/xai.md).
