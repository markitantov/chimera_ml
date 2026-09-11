# Optimizers and schedulers

Built-in optimizers:

- adamw_optimizer (default for direct builder use): lr=1e-3, weight_decay=0.0;
- adam_optimizer: the same defaults;
- sgd_optimizer: lr=1e-2, weight_decay=0.0, momentum=0.0, nesterov=false.

Extra keyword arguments go to the corresponding PyTorch optimizer. The builder
injects the configured model when the factory declares model.

Built-in scheduler keys are steplr_scheduler, cosineannealinglr_scheduler,
and reduceonplateau_scheduler. Their params go to the matching PyTorch
scheduler and the builder injects optimizer.

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
