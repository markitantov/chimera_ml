# Training components

These are the core registered factories used by Trainer. A configuration selects
one loss and may select several compatible metrics; alternative YAML blocks
below are separate configurations, not keys to combine blindly.

## Losses

A loss implements BaseLoss.__call__(output, batch) and returns a differentiable
tensor. Predictions and targets must have compatible shapes.

| Key | Task and defaults |
| --- | --- |
| mse_loss | Regression; reduction=mean, optional sum or none |
| mae_loss | Regression; reduction=mean, optional sum or none |
| ccc_loss | Regression; eps=1e-12; flattens each sample to output features |
| cross_entropy_loss | Class-index classification; label_smoothing=0.0 |
| focal_loss | Class-index multiclass; gamma=2.0, alpha=None, reduction=mean, label_smoothing=0.0 |
| bce_with_logits_loss | Multi-label/multi-hot targets; pos_weight=None, reduction=mean |

Use one of these separate alternatives for regression:

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

For class-index classification:

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

For multi-label or multi-hot targets:

~~~yaml
loss:
  name: bce_with_logits_loss
  params:
    pos_weight: null
    reduction: mean
~~~

The model returns logits for cross entropy, focal, and BCE-with-logits; these
losses do not apply a model-level softmax or sigmoid for you.

## Metrics

Metrics are stateful. Trainer calls reset, update for labeled batches, and
compute at the end of each split/epoch. Regression metrics require matching
floating prediction/target shapes; PRF and confusion metrics use class-index
targets and argmax over the final logits dimension.

Regression metrics:

| Key | Output and parameters |
| --- | --- |
| mae_metric | mae; multioutput=uniform_average or raw_values |
| mse_metric | mse; multioutput=uniform_average or raw_values |
| rmse_metric | rmse; multioutput=uniform_average or raw_values |
| r2_metric | r2; also supports variance_weighted |

~~~yaml
metrics:
  - name: mae_metric
    params:
      multioutput: uniform_average
  - name: mse_metric
    params:
      multioutput: uniform_average
  - name: rmse_metric
    params:
      multioutput: uniform_average
  - name: r2_metric
    params:
      multioutput: variance_weighted
~~~

Classification metrics:

| Key | Output and parameters |
| --- | --- |
| prf_micro_metric | micro precision, recall, f1; zero_division=0 |
| prf_macro_metric | macro precision, recall, f1; zero_division=0 |
| prf_weighted_metric | weighted precision, recall, f1; zero_division=0 |
| confusion_matrix_metric | cm_acc and matrix value; normalize=None, true, pred, or all |

~~~yaml
metrics:
  - name: prf_micro_metric
    params:
      zero_division: 0
  - name: prf_macro_metric
    params:
      zero_division: 0
  - name: prf_weighted_metric
    params:
      zero_division: 0
~~~

~~~yaml
metrics:
  - name: confusion_matrix_metric
    params:
      normalize: "true"
~~~

Do not put MAE/RMSE/R2 and classification PRF into a generic combined task
unless the task and its targets explicitly support both types.

## Optimizers

Factories inject model. Their framework defaults are distinct from any
explicit PyTorch defaults passed through kwargs.

| Key | Constructor | Factory defaults | Required YAML parameters |
| --- | --- | --- | --- |
| adamw_optimizer | torch.optim.AdamW | lr=1e-3, weight_decay=0.0 | none |
| adam_optimizer | torch.optim.Adam | lr=1e-3, weight_decay=0.0 | none |
| sgd_optimizer | torch.optim.SGD | lr=1e-2, weight_decay=0.0, momentum=0.0, nesterov=false | none |

~~~yaml
optimizer:
  name: adamw_optimizer
  params:
    lr: 0.00001
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 0.00000001
~~~

~~~yaml
optimizer:
  name: adam_optimizer
  params:
    lr: 0.001
    weight_decay: 0.0
~~~

~~~yaml
optimizer:
  name: sgd_optimizer
  params:
    lr: 0.01
    momentum: 0.9
    weight_decay: 0.0
    nesterov: true
~~~

## Schedulers

All scheduler factories inject optimizer. The remaining parameters are passed
to the corresponding PyTorch constructor, so required parameters are those
required by that constructor.

| Key | Constructor | Required constructor parameters | Important defaults |
| --- | --- | --- | --- |
| steplr_scheduler | StepLR | step_size | gamma=0.1 |
| cosineannealinglr_scheduler | CosineAnnealingLR | T_max | eta_min=0.0 |
| reduceonplateau_scheduler | ReduceLROnPlateau | none | mode=min, factor=0.1, patience=10 |

~~~yaml
scheduler:
  name: steplr_scheduler
  params:
    step_size: 10
    gamma: 0.1
~~~

~~~yaml
scheduler:
  name: cosineannealinglr_scheduler
  params:
    T_max: 75
    eta_min: 0.0000001
~~~

~~~yaml
scheduler:
  name: reduceonplateau_scheduler
  params:
    mode: min
    factor: 0.1
    patience: 10
~~~

Trainer settings use_scheduler, scheduler_step_per_epoch, and
scheduler_monitor control whether, when, and against which split-prefixed
metric a scheduler is stepped. They are not scheduler constructor parameters.
