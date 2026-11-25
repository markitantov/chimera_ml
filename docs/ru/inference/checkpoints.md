# Чекпоинты и артефакты

`resolve_checkpoints_step` разрешает локальные пути и URL HTTP(S), но не загружает веса в модель. Для локального файла результатом является абсолютный путь. Для URL используется кэш в `work_dir/cache_dir`; существующий файл переиспользуется, если `force_download: false`. Скачивание идёт в `.part`, timeout запроса — 30 секунд.

```yaml
- name: resolve_checkpoints_step
  params:
    checkpoints:
      encoder: /models/encoder.pt
      fusion: https://example.org/fusion.pt
    cache_dir: checkpoints
    force_download: false
```

Следующий prediction step читает `ctx.get_artifact("checkpoints")`, выбирает нужный путь, строит совместимую модель, загружает state dict и переводит модель в `eval()`. Core не знает формат весов и не выполняет эту загрузку автоматически.

`InferenceContext` также хранит `input_path`, `work_dir`, `device`, исходный config и произвольные артефакты. `set_artifact` публикует значение; `get_artifact` читает его. `predictions` — сокращение для артефакта с тем же именем. В параллельном режиме граф объединяет только явно записанные ключи и проверяет владельцев.
