# Установка

## Требования и варианты установки

CHIMERA-ML поддерживает Python `>=3.12,<3.13`. Core объявляет PyTorch `>=2.2,<3.0`, а также NumPy, PyYAML, Typer, tqdm, matplotlib, MLflow, Optuna и requests. CUDA wheel и driver PyTorch выбираются под машину; CHIMERA-ML не устанавливает CUDA driver.

Установка published package в virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install chimera-ml
```

Доступны два равнозначных entry points:

```bash
chimera-ml --help
chimera_ml --help
```

Для разработки репозитория:

```bash
poetry install --with dev,docs
poetry run chimera-ml doctor
```

Группа `docs` нужна только для сборки документации. У example plugins свои зависимости; например:

```bash
python -m pip install -e examples/va_estimation
```

Это устанавливает entry point `chimera_ml.plugins` и делает factories plugin доступными при старте процесса. Датасет и feature files не скачиваются. Их paths нужно подготовить отдельно.

## Проверка окружения

```bash
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list
```

`doctor` показывает Python, PyTorch, доступность CUDA, MLflow, количество registries и plugins. `plugins list` показывает entry points, а `registry list` — реально загруженные keys.

## CPU и CUDA

`train.params.device` обычно принимает `cuda` или `cpu`. Trainer переходит на CPU, если CUDA недоступна. `mixed_precision: true` включает AMP и gradient scaling только на CUDA; на CPU остаётся full precision.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Для GPU нужны совместимые PyTorch CUDA wheel и driver. CPU режим подходит для проверки конфигурации и plugin registration.

## Основные CLI entry points

```text
chimera-ml validate-config --config-path CONFIG.yaml
chimera-ml doctor
chimera-ml train --config-path CONFIG.yaml
chimera-ml eval --config-path CONFIG.yaml --checkpoint-path CHECKPOINT.pt [--with-features]
chimera-ml inference --input INPUT --config-path INFERENCE.yaml [--output OUTPUT.json] [--device auto|cpu|cuda] [--work-dir DIR]
chimera-ml sweep --base-config CONFIG.yaml --sweep-config SWEEP.yaml [--dry-run]
chimera-ml registry list [--type TYPE]
chimera-ml plugins list [--group chimera_ml.plugins]
```

MLflow tracking требует MLflow и local/reachable tracking URI; Optuna нужен для `method: optuna`; Telegram callback дополнительно требует `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID`.
