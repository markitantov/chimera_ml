# Компоненты обучения

Это встроенные зарегистрированные фабрики, используемые Trainer. Конфигурация выбирает одну функцию потерь и совместимые с ней метрики; блоки YAML ниже являются отдельными альтернативами, а не одним объединённым конфигом.

## Функции потерь

Функция потерь измеряет ошибку между предсказанием модели и целевым значением. Во время обучения она предоставляет дифференцируемую целевую функцию, по которой Trainer вычисляет градиенты и обновляет параметры модели.

Функция потерь реализует метод `BaseLoss.__call__(output, batch)` и возвращает дифференцируемый тензор. Формы предсказаний и целевых значений должны соответствовать требованиям выбранной функции потерь.

| Ключ | Задача и значения по умолчанию |
| --- | --- |
| mse_loss | Регрессия; reduction=mean, а также sum/none |
| mae_loss | Регрессия; reduction=mean, а также sum/none |
| ccc_loss | Регрессия; eps=1e-12; разворачивание по признакам результата |
| cross_entropy_loss | Классификация по индексам классов; label_smoothing=0.0 |
| focal_loss | Многоклассовая классификация; gamma=2.0, alpha=None, reduction=mean, label_smoothing=0.0 |
| bce_with_logits_loss | Мультиметочная или multi-hot классификация; pos_weight=None, reduction=mean |

Отдельные варианты для регрессии:

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

Для классификации по индексам классов:

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

Для мультиметочных и multi-hot целевых значений:

~~~yaml
loss:
  name: bce_with_logits_loss
  params:
    pos_weight: null
    reduction: mean
~~~

Модель должна возвращать логиты; эти функции потерь сами не добавляют softmax или sigmoid.

## Метрики

Метрики — это показатели качества, которые описывают работу модели на предсказаниях и целевых значениях. Они накапливают статистику по пакетам и помогают оценивать обучение, но сами по себе не обновляют параметры модели.

Trainer вызывает reset, обновляет состояние на размеченных пакетах и вызывает compute в конце раздела или эпохи.
Метрики регрессии требуют совпадающих форм вещественных значений; PRF и матрица ошибок используют целевые значения с индексами классов и argmax по последней оси логитов.

Регрессия:

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

| Ключ | Параметры и значения по умолчанию |
| --- | --- |
| mae_metric, mse_metric, rmse_metric | multioutput=uniform_average или raw_values |
| r2_metric | также variance_weighted |

Классификация:

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

Ключи PRF возвращают `precision`, `recall` и `f1` для соответствующего способа усреднения. `confusion_matrix_metric` возвращает `cm_acc` и матрицу через `value()`; параметр `normalize` может иметь значение None, true, pred или all. Не объединяйте MAE и PRF классификации в общей задаче без явной поддержки обоих типов целевых значений.

## Оптимизаторы

Оптимизатор получает градиенты функции потерь и обновляет параметры модели. Его параметры определяют правило и величину шага оптимизации, включая скорость обучения и регуляризацию.

Фабрика получает модель через внедрение зависимости. Встроенная фабрика фреймворка:

| Ключ | Конструктор PyTorch | Значения по умолчанию |
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

Планировщик изменяет скорость обучения или другие параметры оптимизатора по мере продвижения обучения. Он помогает управлять размером шага по эпохам или по значению отслеживаемой метрики.

Фабрика планировщика получает оптимизатор через внедрение зависимости; остальные параметры передаются конструктору PyTorch.

| Ключ | Конструктор | Обязательные параметры | Значения по умолчанию |
| --- | --- | --- | --- |
| steplr_scheduler | StepLR | step_size | gamma=0.1 |
| cosineannealinglr_scheduler | CosineAnnealingLR | T_max | eta_min=0.0 |
| reduceonplateau_scheduler | ReduceLROnPlateau | нет | mode=min, factor=0.1, patience=10 |

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

`use_scheduler`, `scheduler_step_per_epoch` и `scheduler_monitor` относятся к Trainer, а не к конструктору планировщика.
