# ORAGEN

## Task and architecture

ORAGEN demonstrates audio-visual gender recognition and age estimation with occlusion-robust multimodal fusion. Audio and multimodal training/evaluation are separate workflows. The shipped video inference uses the plugin model agender_multimodal_model_v3, not a core built-in fusion model.

Fusion returns gender logits followed by one age logit. Inference applies softmax to gender and sigmoid(age_logit) * 100 to age, then averages probabilities and age over windows.

## Install and inspect

~~~bash
python -m pip install -e .
python -m pip install -e examples/oragen
chimera-ml plugins list
chimera-ml registry list --type inference_steps
chimera-ml registry list --type models
~~~

The plugin entry point is oragen = chimera_plugin:register. Relevant files are configs/audio_{train,test}.yaml, configs/multimodal_{train,test}.yaml, configs/inference.yaml, scripts/export_inference_checkpoint.py, and the audio/, fusion/, common/, and inference/ source packages.

Inference uses torchcodec or the plugin ffmpeg fallback, Silero VAD, Ultralytics YOLO, and remote/local Hugging Face weights. The first run needs network access unless all weights are cached.

## Data, caches, and training

Audio configs require features_root, each corpus data_root, labels_file_path, and vad_path. Labels provide gender and age. The dataset reads 16 kHz audio and uses 4-second windows with 2-second shift.

Multimodal configs use VOXCELEB2 and BRAVE-MASKS. Their labels include gender and age and may include mask_type. VAD and media files build windows. features_root contains ORAGEN-style cached records with acoustic_features and visual_features. If absent, configured audio/image feature-extractor checkpoints are needed to create them. The project does not download corpora.

Change features_root, corpora.*.data_root, labels_file_path, vad_path, audio_feature_extractor.checkpoint_path, image_feature_extractor.checkpoint_path, and train_subsets/val_subsets before use. Validate and run paired configs:

~~~bash
chimera-ml validate-config --config-path examples/oragen/configs/audio_train.yaml
chimera-ml train --config-path examples/oragen/configs/audio_train.yaml
chimera-ml eval --config-path examples/oragen/configs/audio_test.yaml \
  --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

~~~bash
chimera-ml validate-config --config-path examples/oragen/configs/multimodal_train.yaml
chimera-ml train --config-path examples/oragen/configs/multimodal_train.yaml
chimera-ml eval --config-path examples/oragen/configs/multimodal_test.yaml \
  --checkpoint-path logs/your_experiment/your_run/checkpoints/last.pt
~~~

Expected artifacts are logs, checkpoints, grouped metrics, optional confusion figures, and snapshots beneath the configured run directory.

## Video inference

The shipped config is sequential:

    input video -> extract_audio_step -> vad_step -> resolve_checkpoints_step
      -> face_detection_step -> build_windows_step -> extract_features_step
      -> fusion_step -> aggregate_windows_step -> print_json_predictions_step

Run it with shipped media:

~~~bash
chimera-ml inference \
  --input examples/oragen/samples/f_25.mp4 \
  --output prediction.json \
  --config-path examples/oragen/configs/inference.yaml \
  --work-dir runs/oragen
~~~

`resolve_checkpoints_step` uses its default `cache_dir: checkpoints`, resolved relative to `--work-dir`, so this command caches audio, image, and fusion weights under `runs/oragen/checkpoints/`. YOLO resolves `yolo26n.pt` separately. CPU is a fallback when CUDA is unavailable. A representative final object uses `gender` (`label`, `class_index`, `probabilities`), `age` (`value`), `num_windows`, `windows` (`index`, `start_sec`, `end_sec`, gender fields, `age_pred`, `num_faces`, `num_frames`), and `debug`. No windows produces null `gender`/`age`, `num_windows` 0, `windows` [], and status `no_predictions`. Probabilities are softmax probabilities and age is a scaled sigmoid output, not calibrated uncertainty.

## XAI and troubleshooting

ORAGEN fusion_step does not expose the on_prediction callback required by xai_step, so XAI is not enabled by default. Use the [XAI guide](../user-guide/xai.md) and [custom inference steps](../inference/custom-steps.md) for the generic mechanism. Common failures are missing plugin/cache/checkpoints, torchcodec/ffmpeg, invalid media, no face/speech, and unwritable work-dir.


## Prediction artifact and JSON envelope

AggregateWindowsStep writes the predictions artifact with gender, age, and
num_windows. When include_windows is true it also contains windows; when
include_debug is true it contains debug.

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

The fusion step creates one window record per feature item. Gender
probabilities are softmax probabilities over the configured class names;
age_pred is sigmoid(age logit) multiplied by age_scale, which is 100. The
aggregate step takes the arithmetic mean of window probabilities and age, then
selects the maximum mean gender probability. It does not report calibrated
confidence.

The built-in JSON writer wraps this artifact as
{"input": "path/to/video.mp4", "predictions": {the artifact above}}. With no
windows the predictions artifact is {"gender": null, "age": null,
"num_windows": 0, "windows": [], "status": "no_predictions"}.
