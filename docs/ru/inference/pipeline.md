# Конфигурация pipeline

Конфигурация инференса отделена от training YAML:

| Поле | Тип и default | Назначение |
| --- | --- | --- |
| `pipeline.name` | строка, `inference_pipeline` | Имя pipeline. |
| `pipeline.parallel` | bool, `false` | Включить DAG-планирование. |
| `runtime.device` | строка, `auto` | Запрошенное устройство. |
| `steps` | список, `[]` | Шаги pipeline. |
| `steps[].id` | строка, по умолчанию `name` | Уникальный id узла. |
| `steps[].after` | строка или список, `[]` | Зависимости в parallel режиме. |
| `steps[].params` | mapping, `{}` | Аргументы фабрики реестра. |

Пример формы:

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

Каждое имя должно быть зарегистрировано в `INFERENCE_STEPS`. `id` меняет только идентификатор узла, а не ключ реестра. В последовательном режиме builder добавляет зависимость от предыдущего шага и не использует явные `after` для планирования. В parallel режиме `after` задаёт DAG; дубли, неизвестные зависимости, self-dependency и циклы отклоняются.

`--device` переопределяет `runtime.device`, `--work-dir` не является полем YAML и задаёт `InferenceContext.work_dir`. `--output` управляет writer только по правилам CLI: в parallel pipeline объявите writer явно и свяжите его через `after`.
