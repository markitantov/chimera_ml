# API метрик

Метрики — накопители с состоянием, работающие в пределах эпохи. `Trainer` вызывает
`reset`, передаёт каждый результат модели и батч в `update`, затем вызывает `compute`
для получения скалярных значений журнала. Функции построения метрик настраиваются через
реестр `METRICS`. См. [Метрики](../user-guide/metrics.md).

## Контракт

::: chimera_ml.metrics.base.BaseMetric

## Метрики регрессии

Метрики регрессии приводят результаты к форме `(N, D)` и поддерживают равномерную
или исходную агрегацию нескольких выходов. R² также поддерживает агрегацию,
взвешенную дисперсией.

::: chimera_ml.metrics.regression_metric.RegressionBaseMetric

::: chimera_ml.metrics.regression_metric.MAEMetric

::: chimera_ml.metrics.regression_metric.mae_metric

::: chimera_ml.metrics.regression_metric.MSEMetric

::: chimera_ml.metrics.regression_metric.mse_metric

::: chimera_ml.metrics.regression_metric.RMSEMetric

::: chimera_ml.metrics.regression_metric.rmse_metric

::: chimera_ml.metrics.regression_metric.R2Metric

::: chimera_ml.metrics.regression_metric.r2_metric

## Метрики классификации

`PRFMetric` возвращает precision, recall и F1 с микро-, макро- или взвешиванием.
`ConfusionMatrixMetric` хранит матрицу классов, возвращает сводную точность через
`compute`, а саму матрицу — через `value`.

::: chimera_ml.metrics.prf_metric.PRFMetric

::: chimera_ml.metrics.prf_metric.prf_micro_metric

::: chimera_ml.metrics.prf_metric.prf_macro_metric

::: chimera_ml.metrics.prf_metric.prf_weighted_metric

::: chimera_ml.metrics.confusion_matrix_metric.ConfusionMatrixMetric

::: chimera_ml.metrics.confusion_matrix_metric.confusion_matrix_metric
