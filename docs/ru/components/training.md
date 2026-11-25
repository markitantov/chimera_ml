# Компоненты обучения

Это встроенные зарегистрированные фабрики, используемые Trainer. Конфигурация
выбирает одну функцию потерь и совместимые метрики; блоки YAML ниже являются отдельными
альтернативами, а не одним объединённым конфигом.

## Функции потерь

Функция потерь реализует BaseLoss.__call__(output, batch) и возвращает differentiable
tensor. Формы predictions и targets должны совпадать.

| Key | Задача и defaults |
| --- | --- |
| mse_loss | Regression; reduction=mean, также sum/none |
| mae_loss | Regression; reduction=mean, также sum/none |
| ccc_loss | Regression; eps=1e-12; flatten по output features |
| cross_entropy_loss | Class-index classification; label_smoothing=0.0 |
| focal_loss | Multiclass classification; gamma=2.0, alpha=None, reduction=mean, label_smoothing=0.0 |
| bce_with_logits_loss | Multi-label/multi-hot; pos_weight=None, reduction=mean |

Отдельные альтернативы для regression:

~~~yaml
loss:
  name: mse_loss
  params:
    reduction: mean
~~~

~~~yaml
loss:
  name: mae_loss
  params:
    reduction: mean
~~~

~~~yaml
loss:
  name: ccc_loss
  params:
    eps: 0.000000000001
~~~

Для class-index classification:

~~~yaml
loss:
  name: cross_entropy_loss
  params:
    label_smoothing: 0.0
~~~

~~~yaml
loss:
  name: focal_loss
  params:
    gamma: 2.0
    alpha: 0.25
    reduction: mean
    label_smoothing: 0.0
~~~

Для multi-label/multi-hot targets:

~~~yaml
loss:
  name: bce_with_logits_loss
  params:
    pos_weight: null
    reduction: mean
~~~

Model должен возвращать logits; эти losses сами не добавляют softmax или
sigmoid.

## Метрики

Trainer вызывает reset, update для labeled batches и compute в конце split/epoch.
Regression metrics требуют matching floating shapes; PRF/confusion используют
class-index targets и argmax по последней оси logits.

Regression:

~~~yaml
metrics:
  - name: mae_metric
    params: {multioutput: uniform_average}
  - name: mse_metric
    params: {multioutput: uniform_average}
  - name: rmse_metric
    params: {multioutput: uniform_average}
  - name: r2_metric
    params: {multioutput: variance_weighted}
~~~

| Key | Parameters/defaults |
| --- | --- |
| mae_metric, mse_metric, rmse_metric | multioutput=uniform_average или raw_values |
| r2_metric | также variance_weighted |

Classification:

~~~yaml
metrics:
  - name: prf_micro_metric
    params: {zero_division: 0}
  - name: prf_macro_metric
    params: {zero_division: 0}
  - name: prf_weighted_metric
    params: {zero_division: 0}
~~~

~~~yaml
metrics:
  - name: confusion_matrix_metric
    params: {normalize: "true"}
~~~

PRF keys возвращают precision/recall/f1 соответствующего average.
confusion_matrix_metric возвращает cm_acc и matrix через value(); normalize
может быть None, true, pred или all. Не объединяйте MAE и classification PRF
в generic task без явной поддержки обоих target types.

## Оптимизаторы

Фабрика получает модель через внедрение зависимости. Встроенная фабрика фреймворка:

| Key | PyTorch constructor | Defaults |
| --- | --- | --- |
| adamw_optimizer | AdamW | lr=1e-3, weight_decay=0.0 |
| adam_optimizer | Adam | lr=1e-3, weight_decay=0.0 |
| sgd_optimizer | SGD | lr=1e-2, weight_decay=0.0, momentum=0.0, nesterov=false |

~~~yaml
optimizer:
  name: adamw_optimizer
  params: {lr: 0.00001, weight_decay: 0.01, betas: [0.9, 0.999], eps: 0.00000001}
~~~

~~~yaml
optimizer:
  name: adam_optimizer
  params: {lr: 0.001, weight_decay: 0.0}
~~~

~~~yaml
optimizer:
  name: sgd_optimizer
  params: {lr: 0.01, momentum: 0.9, weight_decay: 0.0, nesterov: true}
~~~

## Планировщики

Фабрика планировщика получает оптимизатор через внедрение зависимости; остальные params передаются в PyTorch
constructor.

| Key | Constructor | Required parameters | Defaults |
| --- | --- | --- | --- |
| steplr_scheduler | StepLR | step_size | gamma=0.1 |
| cosineannealinglr_scheduler | CosineAnnealingLR | T_max | eta_min=0.0 |
| reduceonplateau_scheduler | ReduceLROnPlateau | none | mode=min, factor=0.1, patience=10 |

~~~yaml
scheduler:
  name: steplr_scheduler
  params: {step_size: 10, gamma: 0.1}
~~~

~~~yaml
scheduler:
  name: cosineannealinglr_scheduler
  params: {T_max: 75, eta_min: 0.0000001}
~~~

~~~yaml
scheduler:
  name: reduceonplateau_scheduler
  params: {mode: min, factor: 0.1, patience: 10}
~~~

use_scheduler, scheduler_step_per_epoch и scheduler_monitor относятся к
Trainer, а не к scheduler constructor.
