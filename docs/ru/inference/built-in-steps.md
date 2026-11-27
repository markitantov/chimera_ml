# Встроенные шаги инференса

Ядро регистрирует resolve_checkpoints_step, write_json_predictions_step,
print_json_predictions_step и xai_step.

## xai_step

xai_step оборачивает шаг предсказания с обратным вызовом и записывает
artifacts["xai"]. Вложенный шаг должен вызывать
on_prediction(model, batch, output) после каждого прямого прохода модели. Его
можно передать объектом или описанием зарегистрированного шага.

Параметры: step, modalities, score_fn, output_mask и ablation_policy. Политика
может быть remove, mask или zero. Вызываемые объекты score_fn и output_mask
передаются только из Python-кода. Полный контракт приведён в руководстве XAI.
