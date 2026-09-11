# Losses

Loss реализует BaseLoss.__call__(output, batch) и возвращает scalar tensor.
Built-in keys: mse_loss/mae_loss для regression, cross_entropy_loss/focal_loss
для class-index classification, bce_with_logits_loss для multi-label targets и
ccc_loss для regression agreement.

~~~yaml
loss:
  name: focal_loss
  params:
    gamma: 2.0
    reduction: mean
~~~

cross_entropy_loss принимает label_smoothing. focal_loss — gamma, scalar или
per-class alpha, reduction и label_smoothing. Regression losses принимают
reduction; ccc_loss — eps и flatten-ит output до batch-by-feature columns.
Shapes targets должны соответствовать выбранному loss.
