# First experiment

This walkthrough uses the repository's VA estimation plugin. It is a real experiment, not a self-contained demo: you need the plugin dependencies, the VA feature/label data, writable output space, and any paths referenced by the YAML.

## 1. Install the runtime and plugin

From the repository root:

```bash
python -m pip install chimera-ml
python -m pip install -e examples/va_estimation
chimera-ml doctor
chimera-ml plugins list
```

The plugin entry point registers `fusion_va_datamodule`, `fusion_model`, `fusion_ccc_mse_loss`, `va_ccc_metric`, and `fusion_framewise_callback`. Confirm the keys in your environment with `chimera-ml registry list`.

## 2. Prepare the data and paths

Edit `examples/va_estimation/configs/multimodal_train.yaml` and set the feature and annotation paths under `data.params`, including the audio and modality feature directories. The plugin expects the VA feature files and labels used by its datamodule; the framework cannot create them. If you keep `telegram_notifier_callback`, export the two Telegram variables or remove that optional callback:

```bash
export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

Do not put credentials in YAML or commit them.

## 3. Validate and train

```bash
chimera-ml validate-config   --config-path examples/va_estimation/configs/multimodal_train.yaml
chimera-ml train   --config-path examples/va_estimation/configs/multimodal_train.yaml
```

Validation is structural. Training additionally requires every plugin key, dataset path, feature shape, and callback monitor to be valid. The example requests CUDA and mixed precision; it falls back to CPU if CUDA is unavailable, but the run will be slower.

## 4. Find logs and checkpoints

The example config sets `log_path: logs` and `experiment_name: va_estimation_fusion`. The CLI supplies the generated run name to the checkpoint and snapshot callbacks. A successful run has this shape:

```text
logs/
└── va_estimation_fusion/
    └── <generated-run-name>/
        ├── train.log
        ├── checkpoints/
        │   ├── last.pt
        │   └── epoch=...pt
        ├── <config-file>.yaml       # snapshot_callback when enabled
        └── code.zip                 # snapshot_callback when enabled
```

The MLflow logger also records metrics and artifacts in the configured tracking store. `checkpoint_callback` saves `model_state_dict`, optimizer state, epoch, global step, and scheduler state when present.

## 5. Evaluate the checkpoint

Use the matching test configuration and a checkpoint from the run:

```bash
chimera-ml eval   --config-path examples/va_estimation/configs/multimodal_test.yaml   --checkpoint-path logs/va_estimation_fusion/your_run/checkpoints/last.pt
```

Evaluation rebuilds the configured datamodule, model, loss, metrics, and callbacks, loads either the callback payload's `model_state_dict` or a raw state dictionary, and requires an exact model architecture match. It evaluates the available train/validation/test loaders exposed by the datamodule; it does not perform deployment inference.

## What to change for your project

Keep the framework-level YAML shape, but replace the plugin-specific `data`, `model`, `loss`, `metrics`, and callback entries with components registered by your project. See [Essential concepts](essential-concepts.md), [dataset preparation](../user-guide/data.md), and [experiment configuration](../user-guide/configuration.md).
