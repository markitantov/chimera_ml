# Объяснимый ИИ (XAI)

CHIMERA-ML предоставляет анализ исключения одной модальности через `explain_batch` и оболочку конвейера `xai_step`. Для каждого образца и модальности выполняется вмешательство, выбирается оценка, а результат сравнивается с базовым предсказанием. Это диагностика локальной чувствительности, а не SHAP, причинный эффект, калиброванная уверенность, точность, надёжность или гарантия доверия.

## Контракт колбэка

`xai_step` оборачивает зарегистрированный шаг предсказания. Тот же шаг должен работать без XAI: колбэк может быть `None`. Точный необязательный параметр и вызов:

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

Колбэк вызывается после исходного прямого прохода `model(batch)` и до прикладной постобработки. `XAIStep` не требует заранее установленной вызываемой функции: он принимает `None`, временно устанавливает свой сборщик, вызывает прежний колбэк, если он был, и восстанавливает значение в `finally`. У оборачиваемого шага должен существовать сам атрибут; иначе оболочка выдаёт `TypeError`.

## Полный зарегистрированный пример

В этом детерминированном примере две модальности. Модель складывает их, поэтому базовое предсказание легко проверить. Один и тот же зарегистрированный шаг можно запускать напрямую или внутри `xai_step`.

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

В YAML можно использовать тот же зарегистрированный ключ, но произвольные вызываемые объекты Python, например `score_fn`, нельзя записать непосредственно в YAML. Их нужно передать из Python-кода или плагина:

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

Без XAI укажите напрямую `name: toy_prediction_step`. `xai_step` есть в реестре ядра, а `toy_prediction_step` — иллюстративный ключ, который приложение или плагин должен зарегистрировать.

## Вмешательства и необработанные объяснения

Поддерживаются явные политики:

- `zero`: заменить тензор выбранной модальности нулями;
- `mask`: отключить канонические маски наличия и времени и при необходимости добавить `<modality>_mask` со значениями `False`;
- `remove`: удалить вход модальности и связанные маски, включая вложенные устаревшие маски.

Выбирайте политику, совместимую с моделью. Отсутствующая модальность получает `already_missing`, неприменимое вмешательство — `unsupported`. В таких записях `ablated`, `delta` и `absolute_delta` равны `null`; это не нулевое влияние. Неожиданные ошибки модели пробрасываются.

По умолчанию напрямую сравнивается `output.preds`: softmax и семантика задачи автоматически не добавляются. `score_fn` должен вернуть тензор, выровненный по батчу. Для каждой валидной позиции:

```text
delta          = baseline_score - ablated_score
absolute_delta = abs(delta)
```

Положительное `delta` означает уменьшение выбранной оценки после абляции, отрицательное — её увеличение. `output_mask_key` или вызываемый объект ограничивает сравнение валидными позициями; используются позиции, валидные и для базового, и для изменённого результата. `validity_fn` выполняется после изменённого прямого прохода и может отклонить образец, но не предотвращает этот проход. `diagnostic_keys` копирует выбранные выровненные по батчу значения из `ModelOutput.aux` в необработанную запись образца.

## Корректные примеры размеров оценки

Временная семантика никогда не выводится автоматически. Первая ось — батч; `temporal_axis` нужно указывать только если выбранная оценка действительно содержит временную ось. Поддерживаются формы `[B]`, `[B, C]`, `[B, T]` и `[B, T, C]`; маски валидности пересекаются поэлементно, а недействительные позиции сериализуются как `null`. Глобальная временная сводка агрегирует сопоставимые наблюдения для каждой позиции выхода.

### Скалярная оценка на образец: `[B]`

Для обычного результата `[B, C]` выберите один компонент:

```python
raw = explain_batch(
    model=model,
    batch=batch,
    score_fn=lambda output: output.preds[:, 0],  # [B]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw)  # без temporal_axis
```

Итоговая оценка имеет форму `[B]` и не содержит временной оси. `temporal_axis=1` здесь некорректен.

### Временная оценка: `[B, T]`

Если модель возвращает одну оценку для каждой временной позиции, оставьте результат в форме `[B, T]`:

```python
raw = explain_batch(model, batch, score_fn=lambda output: output.preds, modalities=["audio", "video"])
summary = summarize_explanations(raw, temporal_axis=1)
```

Здесь ось 0 — батч, ось 1 — время.

### Временная оценка с компонентами: `[B, T, C]`

Для результата `[B, T, C]` выберите компонент, сохранив временную ось:

```python
raw = explain_batch(
    model, batch,
    score_fn=lambda output: output.preds[:, :, 0],  # [B, T]
    modalities=["audio", "video"],
)
summary = summarize_explanations(raw, temporal_axis=1)
```

Выбранная оценка имеет форму `[B, T]`, где ось 1 — время. Если `score_fn` возвращает `[B, T, C]`, `temporal_axis=1` всё равно указывает на время, а `C` остаётся осью компонентов результата. Тензор `[B, C]` по умолчанию не считается временным.

## Минимальная схема артефакта и интерпретация

`explain_batch` возвращает исходный артефакт с полями `ablation_policy`, `modalities` и `samples`. `xai_step` добавляет к нему `summary`. Запись модальности содержит только `status`, необязательный `reason`, `ablated`, знаковую `delta` и `absolute_delta`; выбранные baseline-диагностики копируются из `ModelOutput.aux`. Запись сводки содержит только `mean_signed_delta`, `mean_absolute_delta` и `relative_sensitivity`; порядок модальностей задаёт список `ranking`. Поля версии/схемы, счётчики покрытия, группы равенств, поля доминирующей модальности и автоматические метаданные намеренно отсутствуют.

Относительная чувствительность — описательная величина:

```text
mean_absolute_delta(modality)
-------------------------------
sum(mean_absolute_delta(all comparable modalities))
```

Наблюдения имеют одинаковый вес. Неподдерживаемые и отсутствующие сравнения исключаются, а не превращаются в нули. Временная сводка агрегирует все сопоставимые наблюдения для каждой позиции выхода и не выводит исходные временные метки входа. Артефакт намеренно не содержит полей версии схемы и метода; потребители должны опираться на описанную форму XAI-артефакта.

## Диагностика и временные сводки

`diagnostic_keys` копирует выбранные выровненные по батчу baseline-тензоры из `ModelOutput.aux`; XAI не интерпретирует их как важность. Доступность использует общую семантику масок моделей для входов `[B, D]` и `[B, T, D]`; одномерные пользовательские входы `[B]` считаются присутствующими, а другие ранги явно отклоняются. TFS предоставляет `pair_weights` и `group_weights`, TFS-DCMMOE — адаптивные веса взаимодействий, RAAV — `frame_weights` и `frame_mask`, TOMAR — `rho_<modality>`.

Укажите `temporal_axis=1` для оценок `[B, T]` или `[B, T, C]`. Поддерживаются формы `[B]`, `[B, C]`, `[B, T]` и `[B, T, C]`; недействительные позиции сериализуются как `null` и исключаются из средних.

## Получение и экспорт артефакта XAI

Стандартные средства записи JSON экспортируют только путь к входу и артефакт `predictions`. Они не сериализуют все артефакты `InferenceContext`, поэтому `write_json_predictions_step` автоматически не добавляет `xai`. В Python артефакт можно получить и сохранить явно:

```python
import json
from pathlib import Path

result = build_inference_pipeline(config).run(context)
xai = result.get_artifact("xai")
if xai is None:
    raise ValueError("The pipeline did not produce an xai artifact")
output_path = Path("outputs/explanations.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(xai, indent=2, allow_nan=False), encoding="utf-8"
)
```

Для CLI приложение может зарегистрировать собственный шаг вывода, не изменяя CHIMERA-ML:

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
        path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
        ctx.set_artifact("xai_output_path", path)
        return ctx

@INFERENCE_STEPS.register("write_xai_json_step")
def write_xai_json_step(**params):
    return WriteXAIJsonStep(**params)
```

В последовательном конвейере средство экспорта должно идти после `xai_step`:

```yaml
steps:
  - name: xai_step
    params: {step: {name: toy_prediction_step, params: {}}}
  - name: write_xai_json_step
    params: {output_path: outputs/xai.json}
```

В параллельном конвейере задайте `id` и явную зависимость средства экспорта от XAI (и от узлов подготовки, данные которых оно читает):

```yaml
pipeline: {name: xai-dag, parallel: true}
steps:
  - id: xai
    name: xai_step
    params: {step: {name: toy_prediction_step, params: {}}}
  - id: export
    name: write_xai_json_step
    after: [xai]
    params: {output_path: outputs/xai.json}
```

Это средство экспорта приведено только как документационный пример: перед запуском через CLI его нужно зарегистрировать и установить в приложении или плагине. Добавлять его в исходный код фреймворка не требуется.

## Стоимость и ограничения

Для каждого батча обёрнутый колбэком шаг предсказания выполняет один исходный прямой проход и использует его результат как базовое предсказание XAI. XAI выполняет не более M дополнительных прямых проходов с вмешательством для M выбранных модальностей; отсутствующие модальности могут уменьшить это число. Построение сводки не выполняет прямых проходов. Обычный вызов с M оцениваемыми модальностями включает один исходный прямой проход и ещё M проходов с вмешательством.

Результат зависит от политики абляции, функции оценки, масок, проверок допустимости, режима модели и распределения данных. Эталонные метки не нужны, но измеряется чувствительность к выбранному вмешательству, а не причинный вклад и не качество модели. Разнородные оценки с несовместимыми шкалами нельзя бездумно агрегировать. Временная сводка описывает позиции выхода, а не гарантированные позиции входа. Отсутствующие, неподдерживаемые и недопустимые сравнения должны оставаться видимыми при оценке покрытия.


## Совместимость моделей и маски выходов

Предварительная валидация необязательна: пользовательские модели без неё продолжают работать. Модель с особыми ограничениями интервенций может реализовать `validate_intervention(original, intervened, modality, policy)` и вернуть `None`, если интервенция допустима для всего батча, или строку с причиной неподдерживаемости. Проверка консервативна: смешанный батч отклоняется до частичного forward.

`unsupported` означает, что интервенция не допускает корректного сравнения; `already_missing` означает отсутствие модальности в исходном образце. Числовые поля в обоих случаях равны `null`, а неподдерживаемые строки не участвуют в сводках. Обнуление меняет значения тензора, маскирование меняет маски доступности, а удаление исключает вход — эти политики неэквивалентны. TFS и TFS-DCMMOE требуют сохранения вычислительной ветви и хотя бы одной настроенной направленной пары для каждого поддержанного образца. TOMAR требует anchor-модальность: её удаление или маскирование неподдерживаемо, а обнуление следует существующему контракту модели. RAAV требует хотя бы одной корректной позиции frame.



Чувствительность соответствует определению UDKMO `Delta_m(x) = h(f(x_S)) - h(f(x_(S\{m\})))`: одна и та же модель вычисляет baseline и вмешательство одной модальности, `score_fn` задаёт `h`, а знаковая и абсолютная разности сохраняются. Внутренние диагностики остаются отдельными наблюдениями baseline: TFS и TFS-DCMMOE предоставляют `pair_weights` и `group_weights`, RAAV — `frame_weights` и `frame_mask` (а также attention-тензоры), TOMAR — коэффициенты коррекции `rho_<modality>`. XAI не превращает чувствительность в коэффициенты внимания и не заявляет причинную атрибуцию.

Для временных выходов RAAV явно выберите маску из `ModelOutput.aux` через callable:

```python
raw = explain_batch(
    model, batch,
    score_fn=lambda output: output.preds[..., 0],
    output_mask_key=lambda batch, output: output.aux["frame_mask"],
)
summary = summarize_explanations(raw, temporal_axis=1)
```

Движок пересекает `frame_mask` baseline и intervened; некорректные позиции получают `null` и не участвуют в числовых сводках. Строковые ключи по-прежнему читают маски batch и не переинтерпретируются как ключи `ModelOutput.aux`.
