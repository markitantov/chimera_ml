# Configuration reference

This page describes the framework-level contract. Parameters below are passed
to the selected factory; plugin-defined parameters are not validated by the
core.

## Experiment sections

| Section | Shape | Required | Behavior |
| --- | --- | --- | --- |
| seed | integer | no | Passed to define_seed; defaults to 0. |
| experiment_info | mapping | for train/sweep | params.experiment_name is required; run_name, include_time, datetime_format, and timezone affect run naming. |
| data | name + params | yes for train/eval | Builds a datamodule from DATAMODULES. |
| model | name + params | yes for train/eval | Builds a model from MODELS. |
| train | params | yes for train/eval | Builds TrainConfig. Evaluation changes epochs to 1. |
| loss | name + params | yes for train/eval | Builds a callable from LOSSES. |
| optimizer | name + params | yes for train/eval | Defaults to adamw_optimizer in direct builders. |
| scheduler | name + params | no | Builds a scheduler when present. |
| metrics | list of named items | no | Each item is built from METRICS. |
| callbacks | list of named items | no | Each item is built from CALLBACKS. |
| logging | list of named items | no | Builds configured logger entries. |

A component item uses this form:

~~~yaml
name: registry_key
params:
  plugin_specific_option: value
~~~

## TrainConfig fields

The train.params mapping is passed to TrainConfig:

| Field | Default | Meaning |
| --- | ---: | --- |
| epochs | 10 | Number of training epochs. |
| device | cuda | Requested device; falls back to CPU when CUDA is unavailable. |
| mixed_precision | false | Enables AMP only on CUDA. |
| grad_clip_norm | null | Optional gradient norm clipping. |
| log_every_steps | 50 | MLflow step-log interval. |
| collect_cache | true | Cache eval predictions for callbacks. |
| train_loader_mode | single | single, round_robin, or weighted. |
| train_stop_on | min | Stop multi-loader epochs on the first or last exhausted loader. |
| train_loader_weights | null | Per-loader weights for weighted mode. |
| use_scheduler | false | Enable configured scheduler stepping. |
| scheduler_step_per_epoch | true | Step scheduler per epoch; false steps after train batches. |
| scheduler_monitor | null | Monitor value for metric-driven schedulers. |

logging.console_file_logger.params.log_path is also the root used for sweep
metadata. The train CLI injects experiment_name and generated run_name into
factories where their signatures accept them. Checkpoint and snapshot callbacks
receive run metadata automatically.

YAML values in plugin params are passed as Python keyword arguments. The
plugin factory signature is the authority for component-specific options.
