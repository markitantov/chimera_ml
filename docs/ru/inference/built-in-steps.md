# Встроенные шаги инференса

Актуальный реестр проверяйте в том же окружении, где запускается CLI:

```bash
chimera-ml registry list --type inference_steps
```

Core регистрирует четыре шага. Прикладные extraction, feature, fusion и prediction steps — компоненты плагина, их ключи нельзя выдумывать по имени класса.

## `resolve_checkpoints_step`

Параметры:

| Параметр | Default | Поведение |
| --- | --- | --- |
| `checkpoints` | обязателен | Mapping logical name → local path или URL HTTP(S). |
| `cache_dir` | `checkpoints` | Относительный путь считается от `ctx.work_dir`; абсолютный используется как есть. |
| `force_download` | `false` | Повторно скачать существующий remote cache. |
| `chunk_size` | `8192` | Размер HTTP streaming chunk в байтах. |

```yaml
- name: resolve_checkpoints_step
  params:
    cache_dir: model_cache
    checkpoints:
      fusion: https://example.org/models/fusion.pt
```

Локальный путь должен существовать и быть файлом. Remote download разрешает только HTTP(S), пишет сначала во временный `.part` файл и удаляет его при ошибке; timeout запроса — 30 секунд. Шаг публикует абсолютные пути в `artifacts["checkpoints"]` и каталог в `artifacts["cache_dir"]`. Он не загружает веса в модель.

## `write_json_predictions_step`

`output_path` необязателен. Шаг требует `predictions`, создаёт родительский каталог, пишет JSON с полями `input` и `predictions` с отступом 2 и публикует `artifacts["output_path"]`. Если путь не задан, используется `ctx.input_path` с заменённым suffix `.json`.

## `print_json_predictions_step`

Параметров нет. Шаг требует `predictions` и печатает тот же payload в stdout, не создавая файл.

## `xai_step`

Оборачивает prediction step с атрибутом `on_prediction` и публикует `artifacts["xai"]`. Параметры: обязательный nested `step`; `ablation_policy` (`zero`, `mask`, `remove`); `modalities`; Python-callable `score_fn`; `diagnostic_keys`; `output_mask_key`; `validity_fn`; положительный batch-relative `temporal_axis`. Произвольные callable нельзя выразить YAML — передавайте их из Python или плагина. Подробный контракт и экспорт см. в [XAI](../user-guide/xai.md).
