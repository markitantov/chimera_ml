# Explainable AI (XAI)

CHIMERA-ML предоставляет leave-one-modality-out анализ через `explain_batch` и pipeline-шаг `xai_step`. Для каждого образца и модальности выполняется предсказание с вмешательством, затем сравнивается выбранный score. Это диагностический анализ влияния входа, а не SHAP, доказательство причинности, confidence, accuracy или гарантия надёжности.

## Требования

Модель и все вложенные модули должны быть в `eval()`. Вычисления выполняются в `torch.inference_mode()`. Prediction step должен иметь вызываемый `on_prediction(batch, output, ...)` callback, чтобы XAI мог увидеть каждый batch. Обёртка временно устанавливает callback, вызывает прежний callback и восстанавливает его в `finally`.

## Программный API

```python
from chimera_ml.inference import explain_batch, summarize_explanations

result = explain_batch(
    model=model,
    batch=batch,
    modalities=["audio", "video"],
    ablation_policy="mask",
    score_fn=lambda output: output.preds[:, 0],
    output_mask_key="valid_mask",
)
summary = summarize_explanations(result, temporal_axis=1)
```

`modalities` по умолчанию берутся из `batch.inputs`. Политики: `zero` заменяет tensor нулями; `mask` выключает канонические presence/temporal masks и при необходимости создаёт `<modality>_mask` из `False`; `remove` удаляет input и связанные masks. Отсутствующая модальность получает статус `already_missing`, неподдерживаемая — `unsupported`; их значения не превращаются в нули.

По умолчанию score — `output.preds`. `score_fn` должен вернуть tensor, выровненный по batch. Положительная signed delta означает, что выбранный score уменьшился после абляции (`baseline - ablated`); отрицательная — вырос. `output_mask_key` ограничивает сравнение валидными позициями, а `validity_fn` может отклонить результат после forward, но не предотвращает сам forward. `diagnostic_keys` выбирает batch-aligned значения из `ModelOutput.aux`.

## Результат и summary

Результат версии schema 1 содержит образцы, модальности, baseline и ablated scores, `delta`, `absolute_delta` и статусы; `xai_step` публикует объединённый результат schema version 2 с методом `leave_one_modality_out`, параметрами, batches и summary. Идентификаторы берутся из `batch.meta["sample_ids"]`, иначе генерируются.

`summary` усредняет signed/absolute delta, считает relative influence, ranks, ties и dominant modality только по общим валидным позициям. Missing/unsupported/null не считаются нулевым влиянием. Возможны статусы `insufficient_data`, `no_common_comparison`, `unavailable`, `no_measurable_influence` и другие диагностические статусы.

`temporal_axis` — положительная ось относительно batch для выхода `[B,T]` или `[B,T,C]`; `[B,C]` сам по себе временным не считается. Пример для callback-capable prediction step:

```yaml
- name: xai_step
  params:
    step:
      name: your_prediction_step
      params: {}
    modalities: [audio, video]
    ablation_policy: zero
    temporal_axis: 1
```

YAML не может выразить произвольные `score_fn` и `validity_fn`; задавайте их в Python или внутри плагина. Полный API доступен в [справочнике inference](../api/inference.md).
