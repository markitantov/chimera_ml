# Migration guide

Это не duplicate changelog, а список действий для подтверждённых изменений.

## Config plumbing до 0.2.1

В ветке 0.1.0 удалён yaml_config. Используйте текущие ExperimentConfig/YAML
sections и CLI builders.

## Plugin runtime до 0.2.1

BuildContext добавлен в 0.2.1. Factory, которой нужны metadata предыдущего
component, должна объявить context и использовать register/describe_context,
а не изменять raw YAML или global state.

## Inference output в 0.2.4+

При pipeline.parallel: true --output/-o больше не создаёт
write_json_predictions_step автоматически. Объявите step явно и задайте after.

## Sweep updates

Текущие metadata находятся в
log_path/experiment_name/_sweeps/sweep_id. Grid использует parameters или
trials, Optuna — typed parameters и sweep_target_callback. См.
[Sweeps](../user-guide/sweeps.md) и canonical changelog.

Другие migration actions не добавляются без evidence из source/history.
