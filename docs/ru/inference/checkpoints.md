# Чекпоинты и артефакты

`resolve_checkpoints_step` разрешает локальные пути и URL HTTP(S), но не строит модель и не загружает веса. Для локального файла результат — абсолютный путь. Для URL используется кэш в `work_dir/cache_dir`; существующий файл переиспользуется при `force_download: false`. Скачивание идёт в `.part`, timeout HTTP-запроса — 30 секунд.

```yaml
- name: resolve_checkpoints_step
  params:
    checkpoints:
      encoder: /models/encoder.pt
      fusion: https://example.org/fusion.pt
    cache_dir: checkpoints
    force_download: false
```

Следующий prediction step читает `ctx.get_artifact("checkpoints")`, выбирает logical key, строит совместимую модель, загружает state dict и переводит модель в `eval()`. Формат весов и совместимость модели принадлежат plugin step.

`InferenceContext` содержит `input_path`, `work_dir`, разрешённый `device`, исходный `config` и mapping артефактов. `set_artifact(name, value)` публикует значение и отмечает его как записанное текущим шагом; `get_artifact(name, default)` читает его. `ctx.predictions` — сокращение для артефакта `predictions`.

В parallel режиме pipeline объединяет только ключи, явно записанные через `set_artifact`, и проверяет владельцев. Поэтому стандартный JSON writer экспортирует только `input` и `predictions`, а не весь context и не `xai`; для XAI нужен отдельный output step, описанный в [XAI](../user-guide/xai.md).
