# Исследовательский XAI

'xai_step' оборачивает inference-шаг, предоставляющий необязательный callback:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None

Обычный prediction step остаётся независимым от XAI. Wrapper временно устанавливает callback, вызывает 'explain_batch' для каждого prediction batch, восстанавливает предыдущий callback в 'finally' и публикует результат как artifact 'xai'. Artifact 'predictions' не изменяется.

    steps:
      - name: xai_step
        params:
          step:
            name: fusion_step
            params:
              batch_size: 8
          ablation_policy: zero
          modalities: [audio, image]

Вложенный шаг строится через 'INFERENCE_STEPS' и должен иметь атрибут 'on_prediction'. Без XAI существующая конфигурация шагов работает как раньше.

Поддерживаются явные политики:

- 'zero' — заменить tensor модальности нулями;
- 'mask' — отключить presence/temporal masks;
- 'remove' — удалить ключ модальности из 'Batch.inputs'.

По умолчанию сравниваются исходные 'ModelOutput.preds' без softmax и догадок о типе задачи. Для преобразования используется 'score_fn(preds)'. Результат содержит baseline, ablated, delta = baseline - ablated и absolute delta. Необязательные diagnostics передаются через 'diagnostic_keys'.

Для ORAGEN используется 'zero', поскольку модель требует оба входных ключа и не использует modality masks. Дельты измеряют локальную чувствительность предсказания, но не являются причинными эффектами, SHAP-значениями или калиброванной надёжностью. Оценка качества на отложенной выборке остаётся отдельной функцией 'evaluate_modality_quality'.
