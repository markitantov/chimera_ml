# Метрики

Полный реестр метрик и контракт состояния находятся в разделе [компонентов обучения](../components/training.md).

Метрика реализует сохраняющий состояние интерфейс `BaseMetric`:

~~~python
metric.reset()
metric.update(output, batch)
values = metric.compute()
~~~

Trainer сбрасывает метрики для каждого раздела и эпохи, обновляет их на пакетах с целевыми значениями и добавляет отображение результатов в журналы.

Встроенные ключи: `mae_metric`, `mse_metric`, `rmse_metric`, `r2_metric` для регрессии; `prf_micro_metric`, `prf_macro_metric`, `prf_weighted_metric` для классификации; `confusion_matrix_metric` для матрицы и `cm_acc`.

Метрики регрессии принимают `multioutput` со значениями `raw_values`, `uniform_average` или `variance_weighted`. Метрики PRF принимают `zero_division`. Матрица ошибок поддерживает `normalize` со значениями true, pred, all или без нормализации и возвращает матрицу через `value()`.


## Примеры всех встроенных метрик

    metrics:
      - {name: mae_metric, params: {multioutput: uniform_average}}
      - {name: mse_metric, params: {multioutput: uniform_average}}
      - {name: rmse_metric, params: {multioutput: uniform_average}}
      - {name: r2_metric, params: {multioutput: uniform_average}}
      - {name: prf_macro_metric, params: {zero_division: 0}}
      - {name: confusion_matrix_metric, params: {normalize: true}}

Первые четыре ключа предназначены для регрессии, последние два и `prf_micro_metric`/`prf_weighted_metric` — для классификации по индексам классов. Матрица ошибок не является метрикой регрессии VA.
