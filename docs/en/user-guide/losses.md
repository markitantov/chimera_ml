# Losses

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
