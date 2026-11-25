# Диагностика training

## Configuration и registries

- **Section must be mapping/list**: `data`, `model`, `loss`, `optimizer` — mappings; `metrics`, `callbacks`, `logging` — lists.
- **Нет experiment name**: добавьте `experiment_info.params.experiment_name` для `train`/`sweep`.
- **Unknown registry key**: установите plugin, проверьте `chimera-ml plugins list` и `registry list --type ...`.
- **Collate conflict**: не задавайте одновременно top-level `collate` и `data.params.collate_fn`.

## Data и shapes

- **Dataset path не найден**: `validate-config` не проверяет filesystem. Проверьте paths и current working directory.
- **Нет targets**: training требует `Batch.targets`; исправьте Dataset/collate.
- **Shape mismatch**: сравните feature/target dimensions с model contract. `multimodal_sequence_collate` принимает rank-one `[D]` или rank-two `[T,D]` и consistent feature dimension для каждой modality.
- **Mask mismatch**: temporal masks должны быть `[B,T]`, explicit sample masks — длина unpadded sequence. Missing inputs zero-filled, но presence masks должны отмечать unavailable data.

## Device и numerical errors

- **CUDA unavailable**: `doctor` показывает `cuda_available`; Trainer использует CPU. Для GPU нужны совместимые PyTorch wheel/driver.
- **NaN/Inf**: проверьте normalization, targets, loss scale, learning rate и masks.
- **Mixed precision как CPU**: AMP включается только на resolved CUDA device.

## Monitors, callbacks и output

- **Monitor missing**: используйте точный key `val/loss`, `val/mae` или plugin key `val_framewise/va_ccc_metric`.
- **Нет prediction/figure artifacts**: callbacks требуют MLflow logger и cached labeled outputs; проверьте split selector и `collect_cache`.
- **Telegram fail**: экспортируйте configured token/chat ID env vars; credentials не храните в YAML.

## Checkpoint compatibility

Evaluation делает strict load `model_state_dict`. Нужны тот же model key, constructor params, modality names и feature dimensions. Raw state dict допускается, но не содержит optimizer/scheduler metadata.
