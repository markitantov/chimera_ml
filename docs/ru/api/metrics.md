# API metrics

Metrics — stateful epoch accumulators. \`Trainer\` вызывает \`reset\`, передаёт
каждый model output и batch в \`update\`, затем вызывает \`compute\` для получения
scalar log values. Metric factories настраиваются через \`METRICS\` registry.
См. [Metrics](../user-guide/metrics.md).

## Contract

::: chimera_ml.metrics.base.BaseMetric

## Regression metrics

Regression metrics flatten outputs в \`(N, D)\` и поддерживают uniform или raw
multi-output aggregation. R² также поддерживает variance-weighted output.

::: chimera_ml.metrics.regression_metric.RegressionBaseMetric

::: chimera_ml.metrics.regression_metric.MAEMetric

::: chimera_ml.metrics.regression_metric.mae_metric

::: chimera_ml.metrics.regression_metric.MSEMetric

::: chimera_ml.metrics.regression_metric.mse_metric

::: chimera_ml.metrics.regression_metric.RMSEMetric

::: chimera_ml.metrics.regression_metric.rmse_metric

::: chimera_ml.metrics.regression_metric.R2Metric

::: chimera_ml.metrics.regression_metric.r2_metric

## Classification metrics

\`PRFMetric\` возвращает precision, recall и F1 с micro, macro или weighted
averaging. \`ConfusionMatrixMetric\` хранит class matrix, возвращает summary
accuracy через \`compute\`, а саму matrix — через \`value\`.

::: chimera_ml.metrics.prf_metric.PRFMetric

::: chimera_ml.metrics.prf_metric.prf_micro_metric

::: chimera_ml.metrics.prf_metric.prf_macro_metric

::: chimera_ml.metrics.prf_metric.prf_weighted_metric

::: chimera_ml.metrics.confusion_matrix_metric.ConfusionMatrixMetric

::: chimera_ml.metrics.confusion_matrix_metric.confusion_matrix_metric
