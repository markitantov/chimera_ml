# Affective states recognition

## Task and architecture

This plugin predicts seven emotions (neutral, happy, sad, anger, surprise, disgust, fear) and three sentiments (negative, neutral, positive) from aligned audio, video, and text. The registry model is lefsa_model. Its preds tensor concatenates emotion logits and sentiment logits. Inference applies softmax separately and aggregates window probabilities with duration weights.

    examples/affective_states_recognition/
    ├── configs/{multimodal_train,multimodal_test,inference}.yaml
    ├── samples/
    └── src/
        ├── chimera_plugin.py
        ├── fusion/data, fusion/models
        ├── feature_extractors
        ├── inference (decode, VAD, ASR, faces, fusion)
        └── common (losses, metrics, callback, utilities)

## Install and data

    python -m pip install -e .
    python -m pip install -e examples/affective_states_recognition
    chimera-ml plugins list
    chimera-ml registry list --type datamodules
    chimera-ml registry list --type inference_steps
    chimera-ml validate-config --config-path examples/affective_states_recognition/configs/multimodal_train.yaml

Training requires prepared CMU-MOSEI, MELD, or RAMAS media, labels CSVs, VAD pickles, transcript/ASR columns, and cached feature files. The shipped roots are placeholders: CMUMOSEI uses data/CMUMOSEI/vocals and data/CMUMOSEI/Video_chunk/{subset}; MELD uses data/MELD/vocals/{subset} and data/MELD/videos/{subset}; RAMAS uses data/RAMAS/vocals and data/RAMAS/videos. Label and VAD paths are the corresponding data/labels/* and data/* paths in the YAML.

Set features_root, corpus roots, labels, and VAD paths. Feature mapping is a: wav2vec2, v: emoaffectnet, t: jinaai; windows are 3 seconds maximum, 2-second shift, 2-second minimum, video_target_fps 10, text_max_length 48. There is no dataset download/preprocessing command in this project.

    chimera-ml train --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
    chimera-ml eval --config-path examples/affective_states_recognition/configs/multimodal_test.yaml \\
      --checkpoint-path logs/<experiment>/<run>/checkpoints/last.pt

Training uses emotion_sentiment_loss, combined/emotion/sentiment metrics, AdamW, cosine scheduler, emotion_metrics_callback, checkpoint/snapshot callbacks, and optional console/MLflow logging.

## Video inference

The shipped config is a parallel DAG. Explicit after edges are required so artifact producers finish before consumers:

    extract_audio_step -> vad_step -> build_segments_step -> transcribe_segments_step
    resolve_checkpoints_step -> face_detection_step -> extract_features_step
    extract_features_step -> fusion_step -> write_json_predictions_step
    write_json_predictions_step -> print_json_predictions_step

The implemented stages are audio decoding, Silero VAD, segment construction, Whisper transcription, YOLO face detection, audio/video/text feature extraction, checkpoint-backed fusion, and JSON output.

    chimera-ml inference \\
      --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4 \\
      --output prediction.json \\
      --config-path examples/affective_states_recognition/configs/inference.yaml \\
      --work-dir runs/affective

The config caches emoaffectnet and lefsa; Whisper and YOLO also need external weights. CPU is supported but may be slow. Final output contains emotion and sentiment objects, each with label, class_index, and probabilities, plus num_windows. These are softmax probabilities averaged over windows, not calibrated confidence. Empty windows produce null emotion and sentiment.

    {
      "emotion": {"label": "happy", "class_index": 1, "probabilities": {"neutral": 0.10, "happy": 0.62, "...": 0.28}},
      "sentiment": {"label": "positive", "class_index": 2, "probabilities": {"negative": 0.08, "neutral": 0.17, "positive": 0.75}},
      "num_windows": 2
    }

## XAI and limitations

The example FusionStep performs its own model call and does not declare on_prediction, so xai_step is not currently compatible. Use the generic [XAI guide](../user-guide/xai.md) only when implementing a separate compatible step. No callback was added to this example.
