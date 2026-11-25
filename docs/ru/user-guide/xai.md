# Explainable AI (XAI)

CHIMERA-ML предоставляет leave-one-modality-out анализ через `explain_batch` и обёртку pipeline `xai_step`. Для каждого образца и модальности выполняется вмешательство, выбирается score и результат сравнивается с baseline. Это диагностика локальной чувствительности, а не SHAP, причинный эффект, калиброванная уверенность, accuracy, надёжность или гарантия trust.

## Контракт callback

`xai_step` оборачивает зарегистрированный prediction step. Тот же шаг должен работать без XAI: callback может быть `None`. Точный optional-параметр и вызов:

```python
from collections.abc import Callable
from dataclasses import dataclass
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput

@dataclass
class PredictionStep:
    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = None

    def run(self, ctx):
        model = MyModel().to(ctx.device).eval()
        batch = make_batch(ctx)
        with torch.inference_mode():
            output = model(batch)
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        # Здесь может быть прикладная постобработка.
        ctx.set_artifact("predictions", output.preds.detach().cpu().tolist())
        return ctx
```

Callback вызывается после исходного forward `model(batch)` и до прикладной постобработки. `XAIStep` не требует заранее установленной callable-функции: он принимает `None`, временно устанавливает свой collector, вызывает прежний callback, если он был, и восстанавливает значение в `finally`. У оборачиваемого шага должен существовать сам атрибут; иначе wrapper выдаёт `TypeError`.

## Полный зарегистрированный пример

В этом детерминированном примере две модальности. Модель складывает их, поэтому baseline легко проверить. Один и тот же зарегистрированный шаг можно запускать напрямую или внутри `xai_step`.

```python
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch
from torch import nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference import InferenceConfig, InferenceContext, build_inference_pipeline

class ToyModel(nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        score = batch.inputs["audio"] + batch.inputs["video"]
        return ModelOutput(preds=score)

@dataclass
class ToyPredictionStep:
    on_prediction: Callable[[nn.Module, Batch, ModelOutput], None] | None = None

    def run(self, ctx: InferenceContext) -> InferenceContext:
        model = ToyModel().to(ctx.device).eval()
        batch = Batch(
            inputs={
                "audio": torch.tensor([[2.0]], device=ctx.device),
                "video": torch.tensor([[1.0]], device=ctx.device),
            },
            targets=None,
            meta={"sample_ids": ["toy-0"]},
        )
        with torch.inference_mode():
            output = model(batch)                 # baseline score: [[3.0]]
            if self.on_prediction is not None:
                self.on_prediction(model, batch, output)
        ctx.set_artifact("predictions", output.preds.cpu().tolist())
        return ctx

@INFERENCE_STEPS.register("toy_prediction_step")
def toy_prediction_step(**params):
    return ToyPredictionStep(**params)

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    config = InferenceConfig({
        "pipeline": {"name": "toy-xai"},
        "steps": [{
            "name": "xai_step",
            "params": {
                "step": {"name": "toy_prediction_step", "params": {}},
                "modalities": ["audio", "video"],
                "score_fn": lambda output: output.preds[:, 0],
            },
        }],
    })
    context = InferenceContext(root / "input.bin", root, "cpu", config.raw)
    result = build_inference_pipeline(config).run(context)
    artifact = result.get_artifact("xai")
    assert artifact["samples"][0]["modalities"]["audio"]["delta"] == 2.0
    assert artifact["samples"][0]["modalities"]["video"]["delta"] == 1.0
```

В YAML можно использовать тот же зарегистрированный ключ, но произвольные Python-callable, например `score_fn`, нельзя записать непосредственно в YAML. Их нужно передать из Python-кода или плагина:

```yaml
steps:
  - name: xai_step
    params:
      step:
        name: toy_prediction_step
        params: {}
      modalities: [audio, video]
      ablation_policy: zero
```

Без XAI укажите напрямую `name: toy_prediction_step`. `xai_step` есть в core registry, а `toy_prediction_step` — иллюстративный ключ, который приложение или плагин должен зарегистрировать.

## Вмешательства и raw explanations

Поддерживаются явные политики:

- `zero`: заменить tensor выбранной модальности нулями;
- `mask`: выключить canonical presence/temporal masks и при необходимости добавить `<modality>_mask` со значениями `False`;
- `remove`: удалить input модальности и связанные masks, включая вложенные legacy masks.

Выбирайте политику, совместимую с моделью. Отсутствующая модальность получает `already_missing`, неприменимое вмешательство — `unsupported`. В таких записях `ablated`, `delta` и `absolute_delta` равны `null`; это не нулевое влияние. Неожиданные ошибки модели пробрасываются.

По умолчанию сравнивается `output.preds` напрямую: softmax и task semantics автоматически не добавляются. `score_fn` должен вернуть tensor, выровненный по batch. Для каждой валидной позиции:

```text
delta          = baseline_score - ablated_score
absolute_delta = abs(delta)
```

Положительный delta означает уменьшение выбранного score после абляции, отрицательный — увеличение. `output_mask_key` или callable ограничивает сравнение валидными позициями; используются позиции, валидные и для baseline, и для ablated output. `validity_fn` выполняется после ablated forward и может отклонить sample, но не предотвращает этот forward. `diagnostic_keys` копирует выбранные batch-aligned значения из `ModelOutput.aux` в raw sample.

## Корректные примеры размеров score

Временная семантика никогда не выводится автоматически. Первая ось — batch; `temporal_axis` нужно указывать только если выбранный score действительно содержит временную ось.

### Scalar score на sample: `[B]`

Для обычного output `[B, C]` выберите один компонент:

```python
raw = explain_batch(
    model=model,
    batch=batch,
    score_fn=lambda output: output.preds[:, 0],  # [B]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw)  # без temporal_axis
```

Итоговый score имеет форму `[B]` и не имеет времени. `temporal_axis=1` здесь некорректен.

### Временной score: `[B, T]`

Если модель возвращает один score на каждую временную позицию, оставьте output `[B, T]`:

```python
raw = explain_batch(model, batch, score_fn=lambda output: output.preds, modalities=["audio", "video"])
summary = summarize_explanations(raw, temporal_axis=1)
```

Здесь ось 0 — batch, ось 1 — время.

### Временной score с компонентами: `[B, T, C]`

Для output `[B, T, C]` выберите компонент, сохранив время:

```python
raw = explain_batch(
    model, batch,
    score_fn=lambda output: output.preds[:, :, 0],  # [B, T]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw, temporal_axis=1)
```

Выбранный score имеет `[B, T]`, где axis 1 — время. Если `score_fn` возвращает `[B, T, C]`, `temporal_axis=1` всё равно указывает на время, а `C` остаётся осью output-компонентов. Tensor `[B, C]` по умолчанию не считается временным.

## Схема artifact и интерпретация

`explain_batch` возвращает schema version 1. `xai_step` объединяет batches в artifact `xai` schema version 2: method `leave_one_modality_out`, raw `samples` и аналитический `summary`. Репрезентативный artifact для одного sample:

```json
{
  "schema_version": 2,
  "method": "leave_one_modality_out",
  "ablation_policy": "zero",
  "modalities": ["audio", "video"],
  "samples": [
    {
      "sample_id": "toy-0",
      "baseline": 3.0,
      "modalities": {
        "audio": {"status": "evaluated", "ablated": 1.0, "delta": 2.0, "absolute_delta": 2.0},
        "video": {"status": "evaluated", "ablated": 2.0, "delta": 1.0, "absolute_delta": 1.0}
      },
      "diagnostics": {},
      "metadata": {}
    }
  ],
  "batches": 1,
  "summary": {
    "status": "evaluated",
    "ranking": ["audio", "video"],
    "ties": [],
    "dominant_modality": "audio",
    "dominant_modalities": ["audio"],
    "total_samples": 1,
    "comparable_samples": 1,
    "sample_coverage": 1.0,
    "modalities": {
      "audio": {"mean_signed_delta": 2.0, "mean_absolute_delta": 2.0, "relative_influence": 0.6666666667, "rank": 1, "valid_comparison_count": 1, "common_comparison_count": 1, "comparable_sample_count": 1, "evaluated_sample_count": 1},
      "video": {"mean_signed_delta": 1.0, "mean_absolute_delta": 1.0, "relative_influence": 0.3333333333, "rank": 2, "valid_comparison_count": 1, "common_comparison_count": 1, "comparable_sample_count": 1, "evaluated_sample_count": 1}
    },
    "samples": []
  }
}
```

В реальном artifact могут присутствовать дополнительные поля `availability`, `temporal_axis` и temporal summaries; пример показывает основной контракт и числа. `mean_signed_delta` — среднее signed delta, а `mean_absolute_delta` — среднее абсолютное значение delta. Формула relative influence:

```text
mean_absolute_delta(modality)
--------------------------------
sum(mean_absolute_delta(all comparable modalities))
```

Ranking сортирует модальности по mean absolute delta. `dominant_modality` — единственная модальность с максимальным значением. При равенстве максимальных значений создаётся группа `ties`, `dominant_modality` становится `null`, а все имена попадают в `dominant_modalities`. Если все сравнимые magnitudes равны нулю, status — `no_measurable_influence`, а influence/rank fields равны `null`.

В примере delta `2.0` означает, что удаление audio снизило выбранный score на 2.0; это не означает причинный вклад ровно в две единицы. Чтобы увидеть изменение влияния, изучайте `summary.samples` по каждому sample и, для temporal score, каждую позицию `temporal`. Глобальное summary использует только общие валидные позиции сравнимых модальностей. `valid_comparison_count` специфичен для модальности, `common_comparison_count` — число позиций, оставшихся для совместного сравнения. Missing/unsupported не превращаются в нули.

Глобальная агрегация даёт samples одинаковый вес: сначала для каждого sample считается его собственное среднее signed и absolute delta, затем усредняются эти sample-level значения. `sample_coverage` — доля входных samples с общим сравнением. Поэтому длинная последовательность автоматически не перевешивает короткую.

## Diagnostics и temporal summaries

`diagnostic_keys` выбирает batch-aligned значения из `ModelOutput.aux` и копирует их в raw samples. Это baseline-наблюдения, а не автоматически рассчитанная важность.

Укажите `temporal_axis` в `xai_step` или `summarize_explanations`, если score содержит list-like temporal dimension. `[B, T]` и `[B, T, C]` получают summary по позициям, включая сравнения, ranking и dominant modality. Input-frame, token или segment attribution из этого output sensitivity автоматически не выводится.

## Получение и экспорт XAI artifact

Стандартные JSON writers экспортируют только input path и artifact `predictions`. Они не сериализуют все артефакты `InferenceContext`, поэтому `write_json_predictions_step` автоматически не добавляет `xai`. В Python artifact можно получить и сохранить явно:

```python
import json
from pathlib import Path

result = build_inference_pipeline(config).run(context)
xai = result.get_artifact("xai")
if xai is None:
    raise ValueError("The pipeline did not produce an xai artifact")
Path("outputs/explanations.json").write_text(
    json.dumps(xai, indent=2), encoding="utf-8"
)
```

Для CLI приложение может зарегистрировать собственный output step, не изменяя CHIMERA-ML:

```python
import json
from dataclasses import dataclass
from pathlib import Path
from chimera_ml.core.registry import INFERENCE_STEPS

@dataclass
class WriteXAIJsonStep:
    output_path: str = "xai.json"

    def run(self, ctx):
        artifact = ctx.get_artifact("xai")
        if artifact is None:
            raise ValueError("Expected xai artifact")
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        ctx.set_artifact("xai_output_path", path)
        return ctx

@INFERENCE_STEPS.register("write_xai_json_step")
def write_xai_json_step(**params):
    return WriteXAIJsonStep(**params)
```

В последовательном pipeline exporter должен идти после `xai_step`:

```yaml
steps:
  - name: xai_step
    params: {step: {name: toy_prediction_step, params: {}}}
  - name: write_xai_json_step
    params: {output_path: outputs/xai.json}
```

В parallel pipeline задайте `id` и явную зависимость exporter от XAI (и от узлов подготовки, данные которых он читает):

```yaml
pipeline: {name: xai-dag, parallel: true}
steps:
  - id: xai
    name: xai_step
    after: [prediction]
    params: {step: {name: toy_prediction_step, params: {}}}
  - id: export
    name: write_xai_json_step
    after: [xai]
    params: {output_path: outputs/xai.json}
```

Этот exporter приведён только как документационный пример: перед CLI-запуском его нужно зарегистрировать и установить в приложении/плагине. Добавлять его в source framework не требуется.

## Стоимость и ограничения

Для каждого callback batch XAI выполняет один baseline forward и ещё один forward на каждую выбранную модальность. Для `M` модальностей это примерно `1 + M` forward на batch плюс обычная работа prediction step. На больших моделях и длинных последовательностях это дорого, поэтому выбирайте модальности и batches осознанно.

Результат зависит от ablation policy, score function, masks, validity checks, режима модели и распределения данных. Ground truth не нужен, но измеряется чувствительность к выбранному вмешательству, а не причинный вклад и не качество модели. Разнородные score с несовместимыми шкалами нельзя бездумно агрегировать. Temporal summary описывает output positions, а не гарантированные input locations. Missing, unsupported и invalid comparisons должны оставаться видимыми при оценке coverage.
