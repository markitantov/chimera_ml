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


## Переход от fusion constructors с encoder/head

В feature-only refactor constructors встроенных fusion-моделей изменились.
Ранее они принимали mapping encoders и внешний head; теперь принимают
input_dims, shared_dim и output_dim.

До:

~~~python
TOMARFusionModel(
    encoders=encoders,
    shared_dim=256,
    head=head,
    anchor_modality="text",
)
~~~

После:

~~~python
TOMARFusionModel(
    input_dims={"text": 1024, "audio": 768},
    shared_dim=256,
    anchor_modality="text",
    output_dim=2,
)
~~~

Перенесите precomputed features в Batch.inputs. Теперь model сама выполняет
input projections и prediction layer. Для RAAV также нужны frame_modalities и
context_modality; у TFS-DCMMOE можно задать gate_hidden, у TFS — groups, у
TOMAR — gate_hidden.

Если каждой modality нужны независимые padding и temporal masks, используйте
multimodal_sequence_collate. Для существующих pipelines, которым достаточно
legacy sequence и presence-mask behavior, оставьте masking_collate. Известные
registry keys остаются tfs_fusion_model, tfs_dcmmmoe_fusion_model,
raav_fusion_model и tomar_fusion_model; другие удалённые fusion keys без
подтверждения history не перечисляются.

Другие migration actions не добавляются без evidence из source/history.
