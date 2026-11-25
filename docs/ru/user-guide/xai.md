# Исследовательский XAI

'xai_step' оборачивает inference-шаг, предоставляющий необязательный callback:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None

Обычный prediction step остаётся независимым от XAI. Wrapper временно устанавливает callback, вызывает 'explain_batch' для каждого prediction batch, восстанавливает предыдущий callback в 'finally' и публикует результат как artifact 'xai'. Artifact 'predictions' не изменяется.

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          ablation_policy: zero
          modalities: [x]

Вложенный шаг строится через 'INFERENCE_STEPS'. Чтобы подключить собственный prediction step, добавьте необязательный callback и вызовите его сразу после forward модели:

    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = field(default=None, repr=False)

    output = model(batch)
    if self.on_prediction is not None:
        self.on_prediction(model, batch, output)

Минимальный зарегистрированный шаг, который можно построить из YAML выше:

    from collections.abc import Callable
    from dataclasses import dataclass, field
    import torch
    from torch import nn

    from chimera_ml.core.batch import Batch
    from chimera_ml.core.registry import INFERENCE_STEPS
    from chimera_ml.core.types import ModelOutput

    class ToyModel(nn.Module):
        def forward(self, batch):
            return ModelOutput(batch.inputs["x"])

    @dataclass
    class MyPredictionStep:
        value: float = 1.0
        on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = field(default=None, repr=False)

        def run(self, ctx):
            model = ToyModel().eval()
            batch = Batch({"x": torch.tensor([[self.value]])}, None)
            with torch.inference_mode():
                output = model(batch)
                if self.on_prediction is not None:
                    self.on_prediction(model, batch, output)
            ctx.set_artifact("predictions", output.preds.tolist())
            return ctx

    @INFERENCE_STEPS.register("my_prediction_step")
    def my_prediction_step(**params):
        return MyPredictionStep(**params)

Без XAI шаг настраивается напрямую:

    steps:
      - name: my_prediction_step
        params: {}

С XAI тот же зарегистрированный шаг оборачивается так:

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          modalities: [x]

Без XAI существующая конфигурация шагов работает как раньше.

Политики абляции явные. В режиме 'mask' presence/temporal masks модальности становятся false, а несвязанные masks сохраняются; в режиме 'remove' удаляются input и masks этой модальности. Результаты имеют статусы 'evaluated', 'already_missing' или 'unsupported'; недоступные сравнения представлены null.

Доступны политики:

- 'zero' — заменить tensor модальности нулями;
- 'mask' — отключить presence/temporal masks;
- 'remove' — удалить ключ модальности из 'Batch.inputs'.

По умолчанию сравниваются исходные 'ModelOutput.preds' без softmax и догадок о типе задачи. Для преобразования используется 'score_fn(model_output)'; функция должна вернуть tensor с тем же batch dimension. Результат содержит baseline, ablated, delta = baseline - ablated и absolute delta. Необязательные diagnostics передаются через 'diagnostic_keys'. 'validity_fn' выполняется после ablated forward: она может пометить результат как unsupported, но не может предотвратить этот forward.

Дельты измеряют локальную чувствительность предсказания, но не являются причинными эффектами, SHAP-значениями или калиброванной надёжностью. Оценка качества на отложенной выборке остаётся отдельной функцией 'evaluate_modality_quality'.
