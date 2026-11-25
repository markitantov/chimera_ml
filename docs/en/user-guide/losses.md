# Losses

For the complete loss registry and parameter table, see [Training components](../components/training.md).

A loss implements BaseLoss.__call__(output, batch) and returns a scalar tensor.
Built-in keys are mse_loss and mae_loss for regression,
cross_entropy_loss and focal_loss for class-index classification,
bce_with_logits_loss for multi-label/multi-hot targets, and ccc_loss for
regression agreement.

~~~yaml
loss:
  name: focal_loss
  params:
    gamma: 2.0
    reduction: mean
~~~

cross_entropy_loss accepts label_smoothing. focal_loss accepts gamma, optional
scalar or per-class alpha, reduction, and label_smoothing. Regression losses
accept reduction; ccc_loss accepts eps and flattens outputs to
batch-by-feature columns. Shapes and target semantics must match the loss.


## Complete built-in examples

Use one loss per task output; do not mix regression and class-index classification
targets unless a plugin explicitly supports it.

    loss: {name: mse_loss, params: {reduction: mean}}
    loss: {name: mae_loss, params: {reduction: mean}}
    loss: {name: ccc_loss, params: {eps: 0.000001}}
    loss: {name: cross_entropy_loss, params: {label_smoothing: 0.0}}
    loss: {name: focal_loss, params: {gamma: 2.0, alpha: 0.25, reduction: mean, label_smoothing: 0.0}}
    loss: {name: bce_with_logits_loss, params: {reduction: mean}}

MSE, MAE, and CCC use floating predictions/targets. Cross entropy and focal
loss use class-index targets and logits. BCE-with-logits uses multi-label or
multi-hot targets. The six built-in keys are mse_loss, mae_loss, ccc_loss,
cross_entropy_loss, focal_loss, and bce_with_logits_loss.
