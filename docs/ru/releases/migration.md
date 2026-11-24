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


## Текущие feature-only fusion-модели

Текущие TFS, TFS-DCMMOE, RAAV и TOMAR — это новые feature-only
архитектуры. Они принимают `input_dims`, `shared_dim` и `output_dim`, читают
precomputed features из `Batch.inputs` и сами создают projections и prediction
layers. Это не обратно совместимое продолжение публичного API
`TOMARFusionModel(encoders=..., head=...)`: такого примера нет в выпущенной
реализации `main`, поэтому его нельзя использовать как исторический migration
пример.

Поля конструкторов описаны в руководствах конкретных моделей. RAAV также
требует `frame_modalities` и `context_modality`; у TFS-DCMMOE можно задать
`gate_hidden`, у TFS — `groups`, у TOMAR — `gate_hidden`.

## Намеренное удаление legacy fusion-моделей

В рамках рефакторинга намеренно удалены классы
`FeatureFusionModel`, `PredictionFusionModel`, `GatedFusionModel` и
`GatedPredictionFusionModel`, а также их registry keys
`feature_fusion_model`, `prediction_fusion_model`, `gated_fusion_model` и
`gated_prediction_fusion_model`. Прямого drop-in replacement для этих классов
нет. Выберите одну из четырёх текущих feature-only fusion-моделей, если её
архитектура соответствует задаче, и явно перенесите контракт входов и выходов;
старые параметры конструктора и композиция submodels несовместимы.

Если модальностям нужны независимые padding и temporal masks, используйте
`multimodal_sequence_collate`. Во время CLI `train` и `eval` присутствующая
верхнеуровневая секция `collate:` разрешается через `build_collate()` и
передаётся в factory datamodule как `collate_fn`. Для такой интеграции custom
datamodule должен явно объявить keyword-параметр `collate_fn`. Если секции нет,
CLI не создаёт и не внедряет collate, сохраняя существующий default datamodule,
включая legacy-поведение `MaskingCollate`.

Для удалённого legacy registry key нет прямой замены.
