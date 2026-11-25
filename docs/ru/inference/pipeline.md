# Конфигурация pipeline

Inference YAML отделён от training/evaluation YAML. `InferenceConfig` читает `pipeline`, необязательный `runtime` и список `steps`.

| Поле | Тип и default | Назначение |
| --- | --- | --- |
| `pipeline.name` | строка, `inference_pipeline` | Имя pipeline. |
| `pipeline.parallel` | bool, `false` | Включить DAG-планирование. Некорректный не-bool даёт `TypeError`. |
| `runtime.device` | строка, `auto` | Запрошенное устройство; CLI разрешает его в `cpu` или `cuda`. |
| `steps` | список, `[]` | Шаги pipeline. Каждый mapping требует `name`. |
| `steps[].id` | строка, по умолчанию `name` | Уникальный идентификатор узла. Нужен, когда имя шага повторяется. |
| `steps[].after` | строка или список строк, `[]` | Зависимости в parallel режиме. |
| `steps[].params` | mapping, `{}` | Keyword arguments зарегистрированной factory. |

```yaml
pipeline:
  name: demo
  parallel: false
runtime:
  device: auto
steps:
  - name: my_prediction_step
    params: {}
  - name: write_json_predictions_step
    params:
      output_path: outputs/predictions.json
```

Каждый `name` должен быть ключом `INFERENCE_STEPS`. `id` меняет только идентификатор графа, а не ключ реестра. Необязательный `params` передаётся factory. Не указывайте параметры, которых нет у зарегистрированного компонента.

В последовательном режиме builder добавляет зависимость каждого шага от предыдущего и игнорирует `after` при планировании. В parallel режиме `after` задаёт DAG; duplicate ids, self-dependency, unknown dependency и cycles отклоняются до запуска. Узлы без `after` являются независимыми корнями и могут выполняться одновременно.

`--device` CLI переопределяет `runtime.device`. `--work-dir` не является полем YAML и попадает в `InferenceContext.work_dir`. `--output` автоматически добавляет/переопределяет встроенный writer только в последовательном режиме.
