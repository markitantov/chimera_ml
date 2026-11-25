# Последовательные и параллельные pipeline

## Последовательный режим

При `parallel: false` или отсутствии поля шаги выполняются в порядке списка. Каждый следующий шаг получает артефакты предыдущего. Это естественная цепочка «подготовка → prediction → запись»:

```yaml
pipeline: {name: sequential, parallel: false}
steps:
  - name: prepare_step
  - name: prediction_step
  - name: write_json_predictions_step
```

В этом примере `prepare_step` и `prediction_step` — plugin keys; проверьте их через registry. Встроенными являются только реальные ключи из [каталога](built-in-steps.md).

## Parallel DAG

При `parallel: true` pipeline строит граф. Узлы без незавершённых `after` могут выполняться одновременно в разных потоках. Это параллелизм шагов, а не автоматическое распараллеливание model batches.

```yaml
pipeline:
  name: dag
  parallel: true
steps:
  - id: prepare
    name: prepare_step
  - id: predict
    name: prediction_step
    after: prepare
  - id: diagnostics
    name: diagnostics_step
    after: prepare
  - id: write
    name: write_json_predictions_step
    after: predict
```

Каждый узел получает копию upstream artifacts. При merge учитываются только ключи, записанные через `set_artifact`. Перезапись разрешена downstream-узлу, зависящему от предыдущего владельца; независимая конфликтующая запись даёт `ValueError`. Duplicate ids, unknown dependencies, self-dependency и cycles отклоняются до выполнения.

Для XAI exporter должен зависеть от `xai` node. В последовательном режиме поставьте его после `xai_step`; в DAG задайте `after: [xai]`. Аналогично output writer должен зависеть от шага, который создал `predictions`.
