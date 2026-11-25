# Первый эксперимент

В walkthrough используется VA estimation plugin из репозитория. Это не demo без данных: нужны plugin dependencies, VA feature/label data, writable output directory и paths из YAML.

## 1. Установить runtime и plugin

```bash
python -m pip install chimera-ml
python -m pip install -e examples/va_estimation
chimera-ml doctor
chimera-ml plugins list
```

Plugin регистрирует `fusion_va_datamodule`, `fusion_model`, `fusion_ccc_mse_loss`, `va_ccc_metric` и `fusion_framewise_callback`. Проверяйте keys своей среды через `chimera-ml registry list`.

## 2. Подготовить data и paths

Измените `examples/va_estimation/configs/multimodal_train.yaml`: feature и annotation paths находятся в `data.params`. Plugin ожидает VA feature files и labels; framework их не создаёт. Если оставляете `telegram_notifier_callback`, задайте переменные или удалите callback:

```bash
export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

Не помещайте credentials в YAML или git.

## 3. Validate и train

```bash
chimera-ml validate-config   --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train   --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Validation структурная. Training дополнительно требует plugin keys, dataset paths, feature shapes и существующий callback monitor. Пример просит CUDA и mixed precision; при недоступной CUDA Trainer перейдёт на CPU.

## 4. Найти logs и checkpoints

В примере `log_path: logs`, `experiment_name: va_estimation_fusion`. CLI передаёт generated run name callbacks:

```text
logs/
└── va_estimation_fusion/
    └── <generated-run-name>/
        ├── train.log
        ├── checkpoints/
        │   ├── last.pt
        │   └── epoch=...pt
        ├── <config-file>.yaml
        └── code.zip
```

Config и `code.zip` создаются snapshot callback при включённых параметрах. Checkpoint содержит model/optimizer state, epoch, global step и scheduler state, если scheduler есть. MLflow logger записывает metrics/artifacts в tracking store.

## 5. Оценить checkpoint

```bash
chimera-ml eval   --config-path examples/va_estimation/configs/multimodal_test.yaml   --checkpoint-path logs/va_estimation_fusion/your_run/checkpoints/last.pt
```

Evaluation заново строит datamodule, model, loss, metrics и callbacks, загружает `model_state_dict` из payload или raw state dict и требует точного совпадения архитектуры. Это не deployment inference.

## Для своего проекта

Сохраните framework YAML shape, но замените plugin-specific `data`, `model`, `loss`, `metrics` и callbacks своими зарегистрированными components. См. [Основные понятия](essential-concepts.md), [подготовку datasets](../user-guide/data.md) и [конфигурацию](../user-guide/configuration.md).
