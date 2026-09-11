# Optimizers и schedulers

Built-in optimizers:

- adamw_optimizer (direct builder default): lr=1e-3, weight_decay=0.0;
- adam_optimizer: те же defaults;
- sgd_optimizer: lr=1e-2, weight_decay=0.0, momentum=0.0, nesterov=false.

Дополнительные kwargs передаются PyTorch optimizer. Factory получает model,
если signature явно объявляет model.

Scheduler keys: steplr_scheduler, cosineannealinglr_scheduler и
reduceonplateau_scheduler. Params передаются соответствующему PyTorch
scheduler; builder inject-ит optimizer.

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
