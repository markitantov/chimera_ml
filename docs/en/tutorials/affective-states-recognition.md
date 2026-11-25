# Affective states recognition

## Task and project

This plugin predicts seven emotions and three sentiments from audio, video, and
text. The model registry key is lefsa_model. Its prediction tensor contains
emotion logits followed by sentiment logits. The inference step applies softmax
separately and emits the predictions artifact described below.

## Install and prepare data

~~~bash
python -m pip install -e .
python -m pip install -e examples/affective_states_recognition
chimera-ml plugins list
chimera-ml registry list --type datamodules
chimera-ml registry list --type inference_steps
chimera-ml validate-config --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
~~~

Training requires prepared CMU-MOSEI, MELD, or RAMAS media, label CSV files,
VAD pickles, transcript/ASR columns, and cached feature files. The paths in
the shipped YAML are placeholders. Set features_root, corpus roots, label
paths, VAD paths, and compatible cached features before training.

The configured feature mapping is a: wav2vec2, v: emoaffectnet, and t: jinaai.
Windows are at most 3 seconds, shift by 2 seconds, and require at least
2 seconds. Video features use 10 fps and text uses max length 48. The project
does not download or preprocess these datasets.

~~~bash
chimera-ml train --config-path examples/affective_states_recognition/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/affective_states_recognition/configs/multimodal_test.yaml   --checkpoint-path "logs/your_experiment/your_run/checkpoints/last.pt"
~~~

## Inference DAG

The shipped inference.yaml is a parallel DAG. These are its actual
dependencies:

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

Audio extraction and checkpoint resolution are independent root steps. In the
configuration, face detection explicitly waits for both
resolve_checkpoints_step and transcribe_segments_step. Subsequent steps follow
their declared dependencies: feature extraction waits for face detection; fusion
waits for feature extraction; the writer waits for fusion; printing waits for the
writer. The implemented stages are audio
decoding, Silero VAD, fixed segment construction, Whisper transcription, YOLO
face detection, audio/video/text feature extraction, checkpoint-backed fusion,
and JSON output.

~~~bash
chimera-ml inference   --input examples/affective_states_recognition/samples/01_neutral_neutral_cmumosei.mp4   --output prediction.json   --config-path examples/affective_states_recognition/configs/inference.yaml   --work-dir runs/affective
~~~

The configured pipeline resolves emoaffectnet and lefsa into `model_cache`. Whisper and YOLO
also need their external weights. CPU is supported but can be slow.

## Prediction artifact

Fusion writes this artifact to InferenceContext.predictions:

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

Each segment contains timing, duration, speech duration, the aggregation
weight, transcription, modality-availability flags, window count, and
segment-level emotion/sentiment objects. The weight is constructed as
max(speech duration, segment duration) in seconds by the implementation; since
speech intersections are bounded by the segment, it is normally the segment
duration. Feature extraction averages the audio and video window features
within each segment and uses zero feature tensors when a modality is absent.
The final emotion and sentiment probabilities are weighted averages of segment
probabilities. They are softmax probabilities, not calibrated confidence.

The JSON writer adds an outer envelope:

~~~json
{"input": "path/to/input.mp4", "predictions": {"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}}
~~~

With no features, the predictions artifact is exactly:

~~~json
{"emotion": null, "sentiment": null, "text": "", "segments": [], "status": "no_predictions"}
~~~

Thus num_windows belongs to each segment, not to the top-level predictions
object. Training outputs and inference output are separate workflows.

## XAI and limitations

The example FusionStep performs its own model call and does not expose the
on_prediction callback required by xai_step. XAI is therefore not enabled for
this example. Use the generic XAI and custom inference-step guides when
implementing a separate compatible step. External data, model weights, media
decoding, network access, and writable cache space remain prerequisites.
