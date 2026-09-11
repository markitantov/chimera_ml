# Metrics API

Metrics are stateful epoch accumulators. \`Trainer\` calls \`reset\`, feeds each
model output and batch to \`update\`, then calls \`compute\` to obtain scalar log
values. Configure metric factories through the \`METRICS\` registry. See
[Metrics](../user-guide/metrics.md) for configuration examples.

## Contract

::: chimera_ml.metrics.base.BaseMetric

## Regression metrics

Regression metrics flatten outputs to \`(N, D)\` and support uniform or raw
multi-output aggregation. R² additionally supports variance-weighted output.

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

\`PRFMetric\` reports precision, recall, and F1 with micro, macro, or weighted
averaging. \`ConfusionMatrixMetric\` stores the class matrix and exposes its
accuracy summary through \`compute\` and the matrix through \`value\`.

::: chimera_ml.metrics.prf_metric.PRFMetric

::: chimera_ml.metrics.prf_metric.prf_micro_metric

::: chimera_ml.metrics.prf_metric.prf_macro_metric

::: chimera_ml.metrics.prf_metric.prf_weighted_metric

::: chimera_ml.metrics.confusion_matrix_metric.ConfusionMatrixMetric

::: chimera_ml.metrics.confusion_matrix_metric.confusion_matrix_metric
