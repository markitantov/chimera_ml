# Функции потерь

Полный registry losses и таблица параметров находятся в [Training components](../components/training.md).

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


## Примеры всех встроенных loss

Используйте один согласованный тип targets; regression и class-index
classification не следует смешивать без поддержки plugin.

    loss: {name: mse_loss, params: {reduction: mean}}
    loss: {name: mae_loss, params: {reduction: mean}}
    loss: {name: ccc_loss, params: {eps: 0.000001}}
    loss: {name: cross_entropy_loss, params: {label_smoothing: 0.0}}
    loss: {name: focal_loss, params: {gamma: 2.0, alpha: 0.25, reduction: mean, label_smoothing: 0.0}}
    loss: {name: bce_with_logits_loss, params: {reduction: mean}}

MSE, MAE и CCC используют floating predictions/targets; cross entropy и focal
используют class-index targets и logits; BCE-with-logits — multi-label или
multi-hot targets. Ключи: mse_loss, mae_loss, ccc_loss, cross_entropy_loss,
focal_loss и bce_with_logits_loss.
