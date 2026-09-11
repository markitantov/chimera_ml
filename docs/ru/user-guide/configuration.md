# Configuration reference

Здесь описан framework-level contract. Params передаются выбранной factory;
plugin-defined params core не проверяет.

## Experiment sections

| Section | Shape | Required | Behavior |
| --- | --- | --- | --- |
| seed | integer | no | Передаётся в define_seed; default 0. |
| experiment_info | mapping | для train/sweep | Требует params.experiment_name; run_name и поля времени влияют на имя. |
| data | name + params | да для train/eval | Строит datamodule из DATAMODULES. |
| model | name + params | да для train/eval | Строит model из MODELS. |
| train | params | да для train/eval | Строит TrainConfig; eval выставляет epochs=1. |
| loss | name + params | да для train/eval | Строит callable из LOSSES. |
| optimizer | name + params | да для train/eval | В direct builder default — adamw_optimizer. |
| scheduler | name + params | нет | Строит scheduler при наличии. |
| metrics | список named items | нет | Каждый item строится из METRICS. |
| callbacks | список named items | нет | Каждый item строится из CALLBACKS. |
| logging | список named items | нет | Строит настроенные loggers. |

Форма component:

~~~yaml
name: registry_key
params:
  plugin_specific_option: value
~~~

## TrainConfig fields

train.params передаётся в TrainConfig:

| Field | Default | Meaning |
| --- | ---: | --- |
| epochs | 10 | Число training epochs. |
| device | cuda | Requested device; при отсутствии CUDA — CPU. |
| mixed_precision | false | AMP только на CUDA. |
| grad_clip_norm | null | Optional gradient norm clipping. |
| log_every_steps | 50 | Интервал step-logs MLflow. |
| collect_cache | true | Cache eval predictions для callbacks. |
| train_loader_mode | single | single, round_robin или weighted. |
| train_stop_on | min | Остановка на первом или последнем exhausted loader. |
| train_loader_weights | null | Weights loaders для weighted mode. |
| use_scheduler | false | Включить scheduler stepping. |
| scheduler_step_per_epoch | true | Step per epoch или после train batches. |
| scheduler_monitor | null | Monitor для metric-driven scheduler. |

console_file_logger.params.log_path используется также как root sweep
metadata. CLI inject-ит generated names в factories, если signatures их принимают.
Plugin params должны соответствовать signature factory.
