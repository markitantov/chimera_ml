# Функции потерь

Полный реестр функций потерь и таблица параметров находятся в разделе [компонентов обучения](../components/training.md).

Функция потерь реализует метод `BaseLoss.__call__(output, batch)` и возвращает скалярный тензор. Встроенные ключи: `mse_loss` и `mae_loss` для регрессии, `cross_entropy_loss` и `focal_loss` для классификации по индексам классов, `bce_with_logits_loss` для мультиметочных целевых значений и `ccc_loss` для согласующей регрессии.

~~~yaml
loss:
  name: focal_loss
  params:
    gamma: 2.0
    reduction: mean
~~~

`cross_entropy_loss` принимает `label_smoothing`. `focal_loss` — `gamma`, скалярный или поклассовый `alpha`, `reduction` и `label_smoothing`. Функции потерь регрессии принимают `reduction`; `ccc_loss` — `eps` и разворачивает результат в столбцы формата «пакет × признаки». Формы целевых значений должны соответствовать выбранной функции потерь.


## Примеры всех встроенных функций потерь

Используйте один согласованный тип целевых значений; регрессию и классификацию по индексам классов не следует смешивать без поддержки плагина.

    loss: {name: mse_loss, params: {reduction: mean}}
    loss: {name: mae_loss, params: {reduction: mean}}
    loss: {name: ccc_loss, params: {eps: 0.000001}}
    loss: {name: cross_entropy_loss, params: {label_smoothing: 0.0}}
    loss: {name: focal_loss, params: {gamma: 2.0, alpha: 0.25, reduction: mean, label_smoothing: 0.0}}
    loss: {name: bce_with_logits_loss, params: {reduction: mean}}

MSE, MAE и CCC используют вещественные предсказания и целевые значения; cross entropy и focal используют целевые значения с индексами классов и логиты; BCE-with-logits — многометочные целевые значения в формате multi-hot. Ключи: `mse_loss`, `mae_loss`, `ccc_loss`, `cross_entropy_loss`, `focal_loss` и `bce_with_logits_loss`.
