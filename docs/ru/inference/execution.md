# Последовательные и параллельные pipeline

В последовательном режиме шаги выполняются в порядке списка; каждый следующий получает артефакты предыдущего. Это удобно для цепочки «подготовка → модель → запись».

```yaml
pipeline: {name: sequential}
steps:
  - name: prepare_step
  - name: prediction_step
  - name: write_json_predictions_step
```

В parallel режиме pipeline строит DAG. Готовые узлы без незавершённых `after` могут выполняться одновременно в разных потоках; это параллелизм шагов, а не автоматическое распараллеливание батчей модели.

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
  - id: metrics
    name: metrics_step
    after: prepare
  - id: write
    name: write_json_predictions_step
    after: predict
```

Каждый шаг получает копию артефактов. При слиянии учитываются только ключи, записанные через `set_artifact`. Перезапись допустима для downstream-узла, зависящего от владельца; конфликтующие записи независимых веток дают `ValueError`. Неизвестные зависимости и циклы отклоняются до запуска.
