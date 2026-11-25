# Встроенные шаги инференса

Проверить актуальный реестр можно так:

```bash
chimera-ml registry list --type inference_steps
```

Сейчас core регистрирует четыре шага. Прикладные extraction, fusion и prediction steps обычно приходят из плагинов.

## `resolve_checkpoints_step`

Параметры: обязательный mapping `checkpoints` (логическое имя к локальному пути или URL HTTP(S)); `cache_dir` (`"checkpoints"`); `force_download` (`false`); `chunk_size` (`8192`). Локальный путь должен существовать и быть файлом. Удалённые файлы кэшируются относительно `work_dir`, скачиваются через `.part`, а при ошибке временный файл удаляется. Шаг записывает `artifacts["checkpoints"]` и `artifacts["cache_dir"]`, но не создаёт модель и не загружает state dict.

```yaml
- name: resolve_checkpoints_step
  params:
    cache_dir: model_cache
    checkpoints:
      fusion: https://example.org/models/fusion.pt
```

## `write_json_predictions_step`

`output_path` необязателен. Шаг требует артефакт `predictions`, записывает `{"input": ..., "predictions": ...}` с отступами 2 и публикует `artifacts["output_path"]`. Без `output_path` используется входной путь с суффиксом `.json`; родительский каталог создаётся.

## `print_json_predictions_step`

Не имеет параметров, требует `predictions` и печатает тот же JSON в stdout. Файл не создаёт.

## `xai_step`

Оборачивает prediction step с callback `on_prediction` и публикует `artifacts["xai"]`. Параметры: обязательный `step`; `ablation_policy` (`zero`, `mask`, `remove`); `modalities`; Python-callable `score_fn`; `diagnostic_keys`; `output_mask_key`; `validity_fn`; положительный batch-relative `temporal_axis`. YAML не может выразить произвольную Python-функцию, поэтому callable обычно задаётся программно или плагином. Полный контракт — в [XAI](../user-guide/xai.md).
