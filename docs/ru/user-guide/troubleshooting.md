# Диагностика обучения

## Конфигурация и реестры

- **Раздел должен быть отображением или списком**: `data`, `model`, `loss`, `optimizer` — mappings; `metrics`, `callbacks`, `logging` — lists.
- **Нет experiment name**: добавьте `experiment_info.params.experiment_name` для `train`/`sweep`.
- **Неизвестный ключ реестра**: установите plugin, проверьте `chimera-ml plugins list` и `registry list --type ...`.
- **Collate conflict**: не задавайте одновременно top-level `collate` и `data.params.collate_fn`.

## Данные и формы

- **Путь к датасету не найден**: `validate-config` не проверяет filesystem. Проверьте paths и current working directory.
- **Нет targets**: training требует `Batch.targets`; исправьте Dataset/collate.
- **Несовпадение формы**: сравните feature/target dimensions с model contract. `multimodal_sequence_collate` принимает rank-one `[D]` или rank-two `[T,D]` и consistent feature dimension для каждой modality.
- **Несовпадение маски**: temporal masks должны быть `[B,T]`, explicit sample masks — длина unpadded sequence. Missing inputs zero-filled, но presence masks должны отмечать unavailable data.

## Устройство и численные ошибки

- **CUDA unavailable**: `doctor` показывает `cuda_available`; Trainer использует CPU. Для GPU нужны совместимые PyTorch wheel/driver.
- **NaN/Inf**: проверьте normalization, targets, loss scale, learning rate и masks.
- **Mixed precision как CPU**: AMP включается только на resolved CUDA device.

## Мониторы, колбэки и результаты

- **Monitor missing**: используйте точный key `val/loss`, `val/mae` или plugin key `val_framewise/va_ccc_metric`.
- **Нет prediction/figure artifacts**: callbacks требуют MLflow logger и cached labeled outputs; проверьте split selector и `collect_cache`.
- **Telegram fail**: экспортируйте configured token/chat ID env vars; credentials не храните в YAML.

## Совместимость контрольных точек

Evaluation делает strict load `model_state_dict`. Нужны тот же model key, constructor params, modality names и feature dimensions. Raw state dict допускается, но не содержит optimizer/scheduler metadata.
