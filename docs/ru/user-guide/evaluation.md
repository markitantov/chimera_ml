# Оценка checkpoint

Validation в `Trainer.fit` выполняется после каждого train epoch и используется callbacks. Standalone `eval` отвечает на вопрос о сохранённой модели, не делает optimizer updates и может запускаться с отдельной test configuration.

```bash
chimera-ml eval   --config-path examples/va_estimation/configs/multimodal_test.yaml   --checkpoint-path logs/va_estimation_fusion/<run>/checkpoints/last.pt
```

Обе options обязательны. `--with-features` дополнительно требует features из `ModelOutput.aux["features"]`:

```bash
chimera-ml eval -c test.yaml --checkpoint-path checkpoints/best.pt --with-features
```

CLI строит те же datamodule/model/loss/metrics/optimizer/callbacks и загружает checkpoint на CPU с `weights_only=True`. Допустим payload с `model_state_dict` или raw state dict. `load_state_dict(strict=True)` требует точного совпадения model architecture, modality names и dimensions.

Train/validation/test loaders объединяются; mapping names sanitizеd, а unnamed loaders получают split prefix. Поэтому keys выглядят как `val/loss`, `val/mae`, `test2/rmse`. Evaluation вызывает callback hooks вокруг одного epoch. При `collect_cache` сохраняются predictions, targets, metadata и optional features для callbacks.

Artifacts зависят от configured callbacks/loggers: local logs находятся в run directory, MLflow получает metrics и CSV/figure/config artifacts. Evaluation — не deployment inference и не dataset-level XAI.

Ошибки `--with-features` означают, что нет `aux["features"]` с batch dimension, а checkpoint errors обычно означают несовпадение config и model, создавшей файл.
