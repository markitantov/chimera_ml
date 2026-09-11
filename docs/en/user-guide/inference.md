# Inference

Inference is a separate YAML-driven pipeline:

~~~bash
chimera-ml inference \
  --input video.mp4 \
  --output prediction.json \
  --config-path examples/oragen/configs/inference.yaml \
  --device auto \
  --work-dir work
~~~

The CLI accepts device auto, cpu, or cuda. auto selects CUDA when available,
otherwise CPU. If work-dir is omitted, a temporary directory is created.

## Config shape

~~~yaml
pipeline:
  name: demo
  parallel: false

runtime:
  device: auto

steps:
  - name: extract_audio_step
    params:
      sample_rate: 16000
  - name: print_json_predictions_step
~~~

Every step is resolved from INFERENCE_STEPS and implements
run(ctx: InferenceContext) -> InferenceContext. The context has input_path,
work_dir, device, raw config, and shared artifacts. Use set_artifact and
get_artifact; predictions is a convenience property for the predictions key.

## Sequential and DAG modes

With parallel false or omitted, steps run in config order and each depends on
the previous step. With parallel true, after is authoritative: steps without
dependencies are roots and can start together. Step ids default to name;
repeated names need explicit id values. Unknown dependencies, self-dependencies,
duplicate ids, and cycles raise ValueError.

Each parallel step receives a copy of current artifacts. Results merge using
keys written through set_artifact. A branch may overwrite an upstream artifact
when it depends on that upstream node; unrelated branches may not write the
same key.

## Checkpoints and output

resolve_checkpoints_step accepts local files or http/https references, writes
remote files to a work-dir-relative cache, and exposes absolute paths in
artifacts["checkpoints"]. Cached files are reused unless force_download is true.
Failed downloads remove partial files.

write_json_predictions_step writes input and predictions JSON; the print step
prints the same payload. The CLI --output can create or override the write
step in sequential mode. In parallel mode it cannot auto-create it: declare
the step explicitly and set after dependencies in YAML.
