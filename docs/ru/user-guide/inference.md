# Inference

Inference — отдельный YAML-driven pipeline:

~~~bash
chimera-ml inference \
  --input video.mp4 \
  --output prediction.json \
  --config-path examples/oragen/configs/inference.yaml \
  --device auto \
  --work-dir work
~~~

CLI принимает device auto, cpu или cuda. auto выбирает CUDA при доступности,
иначе CPU. Без work-dir создаётся temporary directory.

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

Каждый step строится из INFERENCE_STEPS и реализует
run(ctx: InferenceContext) -> InferenceContext. Context содержит input_path,
work_dir, device, raw config и shared artifacts; используйте set_artifact и
get_artifact.

## Sequential и DAG modes

При parallel false или отсутствии поля steps выполняются в config order и
зависят от предыдущего. При parallel true authoritative является after:
steps без dependencies — root nodes и могут стартовать одновременно. Step id
по умолчанию равен name; повторяющимся names нужны explicit id. Unknown
dependencies, cycles, duplicate ids и self-dependencies дают ValueError.

Каждый parallel step получает copy текущих artifacts. Merge учитывает keys,
записанные через set_artifact. Unrelated branches не могут записать один key;
зависимая downstream branch может переопределить upstream artifact.

## Checkpoints и output

resolve_checkpoints_step принимает local files или http/https refs, сохраняет
remote files в work-dir-relative cache и публикует absolute paths в
artifacts["checkpoints"]. Cache переиспользуется без force_download; partial
files удаляются при ошибке.

write_json_predictions_step пишет input/predictions JSON, print step печатает
тот же payload. CLI --output создаёт/переопределяет write step только в
sequential mode; в parallel mode step нужно объявить явно и задать after.
