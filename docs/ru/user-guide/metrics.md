# Metrics

Полный registry metrics и state contract находятся в [Training components](../components/training.md).

Metric реализует stateful interface BaseMetric:

~~~python
metric.reset()
metric.update(output, batch)
values = metric.compute()
~~~

Trainer сбрасывает metrics для каждого split/epoch, обновляет их на batches с
targets и добавляет mapping в logs.

Built-in keys: mae_metric, mse_metric, rmse_metric, r2_metric для regression;
prf_micro_metric, prf_macro_metric, prf_weighted_metric для classification;
confusion_matrix_metric для matrix и cm_acc.

Regression metrics принимают multioutput raw_values, uniform_average или
variance_weighted. PRF metrics принимают zero_division. Confusion matrix
поддерживает normalize true, pred, all или отсутствие normalization и отдаёт
matrix через value().


## Примеры всех встроенных metrics

    metrics:
      - {name: mae_metric, params: {multioutput: uniform_average}}
      - {name: mse_metric, params: {multioutput: uniform_average}}
      - {name: rmse_metric, params: {multioutput: uniform_average}}
      - {name: r2_metric, params: {multioutput: uniform_average}}
      - {name: prf_macro_metric, params: {zero_division: 0}}
      - {name: confusion_matrix_metric, params: {normalize: true}}

Первые четыре ключа предназначены для regression, последние два и
prf_micro_metric/prf_weighted_metric — для class-index classification.
Confusion matrix не является метрикой VA regression.
