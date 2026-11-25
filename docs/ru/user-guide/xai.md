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

Дельты измеряют локальную чувствительность предсказания, но не являются причинными эффектами, SHAP-значениями или калиброванной надёжностью.


## Анализ влияния modality

После raw explanations 'XAIStep' добавляет детерминированный 'summary' в artifact 'xai'. Summary работает только с уже собранными samples и не запускает модель повторно.

Для каждой сравнимой modality:

- 'mean_signed_delta = mean(delta)' показывает направление изменения выбранного score;
- 'mean_absolute_delta = mean(abs(delta))' показывает измеренную чувствительность;
- 'relative_influence = mean_absolute_delta / sum(mean_absolute_delta)' по общему набору сравнимых modality;
- 'rank' сортирует modality по убыванию mean absolute delta.

Для общего ranking используются только common valid output positions. 'null', 'already_missing', 'unsupported' и недостаточные сравнения не считаются нулевым влиянием. В глобальной агрегации samples имеют одинаковый вес: сначала считается среднее внутри sample, затем среднее по samples. Relative influence нормализуется после агрегации абсолютных значений.

Ties перечислены в 'ties'; 'dominant_modality' равен 'null' при tie и при нулевом измеренном влиянии. Evaluated delta, равный нулю, отличается от отсутствующей или unsupported modality.

Сокращённая форма artifact:

    {
      "schema_version": 2,
      "method": "leave_one_modality_out",
      "ablation_policy": "zero",
      "modalities": ["audio", "video"],
      "summary": {
        "status": "evaluated",
        "ranking": ["audio", "video"],
        "dominant_modality": "audio",
        "dominant_modalities": ["audio"],
        "total_samples": 2,
        "comparable_samples": 2,
        "sample_coverage": 1.0,
        "modalities": {
          "audio": {
            "status": "evaluated",
            "mean_signed_delta": 0.3,
            "mean_absolute_delta": 0.3,
            "relative_influence": 0.75,
            "rank": 1,
            "valid_comparison_count": 2,
            "common_comparison_count": 2
          },
          "video": {
            "status": "evaluated",
            "mean_signed_delta": -0.1,
            "mean_absolute_delta": 0.1,
            "relative_influence": 0.25,
            "rank": 2,
            "valid_comparison_count": 2,
            "common_comparison_count": 2
          }
        },
        "samples": []
      },
      "samples": []
    }

Raw 'samples' сохраняются рядом с 'summary'. При 'delta = baseline - ablated' положительный signed delta означает падение выбранного score после ablation, отрицательный — рост. Например, audio delta '0.3' означает падение на '0.3', а text delta '-0.2' — рост на '0.2'. Missing и unsupported записи сохраняют status и получают null-поля влияния.

Для явного temporal output можно задать 'temporal_axis' на 'xai_step' относительно batch-aligned score:

    steps:
      - name: xai_step
        params:
          step:
            name: my_prediction_step
            params: {}
          modalities: [audio, video]
          temporal_axis: 1

Так можно анализировать '[B, T]' и '[B, T, C]' по output positions. '[B, C]' интерпретируется как temporal только при явном 'temporal_axis: 1'; по умолчанию ни одна размерность не считается временем. Это sensitivity output, а не attribution отдельных frame/token/segment входа. Идентификаторы окон и timing metadata сохраняются только если их передало приложение.

'score_fn(model_output)' должен возвращать сравнимые значения. Heterogeneous output components с разными scale нельзя осмысленно агрегировать; сначала выберите или преобразуйте score. Influence относится к выбранным intervention и score. Это не probability, causal contribution, confidence, accuracy, reliability или trust score. Ground truth не нужен, summary не добавляет forward passes.
