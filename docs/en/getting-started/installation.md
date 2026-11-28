# Installation

## Requirements and installation choices

CHIMERA-ML supports Python `>=3.12`. The core project declares PyTorch `>=2.4` on Python 3.12, `>=2.6` on Python 3.13, and `>=2.9` on Python 3.14, plus NumPy, PyYAML, Typer, tqdm, matplotlib, MLflow, Optuna, and requests. PyTorch's CUDA wheel and driver must be selected for the target machine; CHIMERA-ML does not install a CUDA driver.

Install the published package into a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install chimera-ml
```

The two installed entry points are equivalent:

```bash
chimera-ml --help
chimera_ml --help
```

For repository development, install the Poetry groups used by the project:

```bash
poetry install --with dev,docs
poetry run chimera-ml doctor
```

The `docs` group is only needed to build documentation. The example plugins have their own dependencies and should be installed from their directories, for example:

```bash
python -m pip install -e examples/va_estimation
```

This installs the plugin's entry point (`chimera_ml.plugins`) and makes its datamodule/model/loss/metric/callback factories available at process startup. It does not download the VA dataset or feature files. Follow the path requirements in the example configuration and README before training.

## Verify the environment

Run these safe checks after installation:

```bash
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list
```

`doctor` reports Python, PyTorch, CUDA availability, MLflow, registry counts, and discovered plugins. `plugins list` shows entry points; `registry list` shows the keys actually loaded in the current environment.

## Dependency reproducibility

The version ranges in package metadata describe supported installations. `poetry.lock` records the reproducible development environment and is used by CI in locked mode. Scientific experiments should preserve their own environment lock or export alongside checkpoints; the development lock is not a replacement for an experiment snapshot. CI also checks lowest-direct and latest resolved dependency sets independently of the lock.

## CPU and CUDA

Training's `train.params.device` accepts the requested device string used by `TrainConfig`, normally `cuda` or `cpu`. The Trainer resolves to CPU when CUDA is unavailable. `mixed_precision: true` enables AMP and gradient scaling only when the resolved device is CUDA; on CPU it remains full precision.

Check the actual installation before choosing CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

A CUDA-capable PyTorch wheel and a compatible driver are prerequisites for GPU execution. A CPU run is still a supported way to validate configuration and plugin registration.

## Main CLI entry points

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

Optional infrastructure is explicit: MLflow tracking needs the installed MLflow dependency and a local or reachable tracking URI; Optuna is required for `method: optuna` sweeps; Telegram notifications additionally need `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
