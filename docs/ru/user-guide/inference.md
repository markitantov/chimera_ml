# Inference

Inference — отдельный конфигурируемый execution pipeline. Он не вызывает `Trainer.evaluate`: создаёт `InferenceContext`, запускает registered steps и передаёт между ними именованные artifacts.

~~~bash
chimera-ml inference -i input.mp4 -o prediction.json -c inference.yaml --device auto --work-dir work
~~~

`--input` и `--config-path` обязательны. `--device`: `auto`, `cpu` или `cuda`; `auto` выбирает CUDA при наличии. Если `--work-dir` не задан, CLI создаёт временную папку.

## Конфигурация

~~~yaml
pipeline:
  name: demo
  parallel: false
runtime:
  device: auto
steps:
  - name: resolve_checkpoints_step
    params:
      cache_dir: checkpoints
      checkpoints:
        model: /absolute/path/to/model.pt
  - name: my_prediction_step
    params:
      checkpoint_key: model
  - name: write_json_predictions_step
    params:
      output_path: prediction.json
~~~

Каждый шаг содержит registry `name`, необязательные `id`, `after` и `params`. Контракт шага: `run(ctx: InferenceContext) -> InferenceContext`. Шаг читает `input_path`, `device` и artifacts, а результаты публикует через `ctx.set_artifact`.

## Встроенные steps

| Key | Параметры | Входы и выходы |
| --- | --- | --- |
| `resolve_checkpoints_step` | `checkpoints`, `cache_dir="checkpoints"`, `force_download=false`, `chunk_size=8192` | Использует local или HTTP(S) references; пишет абсолютные пути в `artifacts["checkpoints"]` и cache в `artifacts["cache_dir"]`. |
| `write_json_predictions_step` | `output_path=null` | Требует `predictions`, записывает `input` и `predictions` в JSON и публикует `output_path`. |
| `print_json_predictions_step` | нет | Требует `predictions` и печатает тот же JSON без файла. |
| `xai_step` | вложенный `step`, `modalities`, `ablation_policy` и optional score/diagnostics | Оборачивает совместимый prediction step и публикует `xai`; см. [XAI](xai.md). |

Prediction, feature-extraction и model-specific steps из examples являются plugin components. Проверьте текущие регистрации командой `chimera-ml registry list --type inference_steps`.

## Sequential и parallel режимы

При `parallel: false` или без этого поля шаги выполняются по порядку. CLI `--output` может создать или заменить `write_json_predictions_step` только в этом режиме. При `parallel: true` шаги становятся DAG nodes, а `after` задаёт зависимости. Шаги без `after` — roots; для повторяющихся names нужны явные `id`.

Unknown dependencies, duplicate IDs, cycles и конфликты записи одного artifact являются ошибками. В parallel mode output step нужно объявить в YAML и указать его dependencies.

## Checkpoints и artifacts

`resolve_checkpoints_step` повторно использует cache, если `force_download: true` не задан. Неполные downloads удаляются. Downstream plugin steps должны использовать тот же checkpoint key в `artifacts["checkpoints"]`. JSON создаётся только явным output step.
