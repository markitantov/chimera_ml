# Troubleshooting

## Config section must be a mapping/list

Причина: ExperimentConfig.validate enforces framework shape.

Решение: data, model, train, loss и optimizer должны быть mappings; metrics,
callbacks и logging — lists элементов {name, params}. Снова запустите
chimera-ml validate-config.

## Unknown registry key

Причина: factory не зарегистрирована; обычно plugin не установлен или entry
point/import завершился ошибкой.

Решение: проверьте chimera-ml plugins list и registry list --type для нужного
типа, затем entry point и warnings.

## Training batch без targets

Trainer требует targets для optimization. Используйте labeled dataset для
training; для inference применяйте inference pipeline.

## Checkpoint/output errors в inference

Local checkpoint ref должен быть file, remote ref — доступным http/https URL.
В parallel mode --output не создаёт output step автоматически. Проверьте path и
объявите write_json_predictions_step с after dependencies.

## Callback не находит monitor

Monitor — точный key в flat epoch logs. Используйте val/loss или точный
plugin metric key; тот же key нужен sweep_target_callback.

## with_features fails

Model output не содержит aux["features"]. Добавьте tensor или вызовите
Trainer.evaluate напрямую с feature_extractor.
