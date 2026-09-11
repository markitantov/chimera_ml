# Metrics

A metric implements the stateful BaseMetric interface:

~~~python
metric.reset()
metric.update(output, batch)
values = metric.compute()
~~~

The trainer resets metrics for each split/epoch, updates them for batches with
targets, and merges the returned mapping into logs.

Built-in keys are mae_metric, mse_metric, rmse_metric, and r2_metric for
regression; prf_micro_metric, prf_macro_metric, and prf_weighted_metric for
class-index classification; and confusion_matrix_metric for a matrix plus
cm_acc.

Regression metrics accept multioutput: raw_values, uniform_average, or
variance_weighted. PRF metrics accept zero_division. The confusion matrix
accepts normalization true, pred, all, or no normalization and exposes the
matrix through value().

Metrics return scalar-compatible values for trainer logs. Plugin metrics may
return several named values; callbacks can monitor exact keys such as val/loss.
