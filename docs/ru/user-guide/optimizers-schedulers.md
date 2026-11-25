# Оптимизаторы и планировщики

Полный каталог фабрик оптимизаторов и планировщиков находится в разделе [компонентов обучения](../components/training.md).

Встроенные оптимизаторы:

- `adamw_optimizer` (значения по умолчанию непосредственного строителя): `lr=1e-3`, `weight_decay=0.0`;
- `adam_optimizer`: те же значения по умолчанию;
- `sgd_optimizer`: `lr=1e-2`, `weight_decay=0.0`, `momentum=0.0`, `nesterov=false`.

Дополнительные именованные аргументы передаются оптимизатору PyTorch. Фабрика получает модель, если её сигнатура явно объявляет параметр `model`.

Ключи планировщиков: `steplr_scheduler`, `cosineannealinglr_scheduler` и `reduceonplateau_scheduler`. Параметры передаются соответствующему планировщику PyTorch; строитель внедряет `optimizer`.

~~~yaml
optimizer:
  name: adamw_optimizer
  params:
    lr: 0.00001
    weight_decay: 0.01

scheduler:
  name: cosineannealinglr_scheduler
  params:
    T_max: 75
    eta_min: 0.0000001

train:
  params:
    use_scheduler: true
    scheduler_step_per_epoch: true
~~~
