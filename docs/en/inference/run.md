# Run inference

This guide uses a generic plugin prediction step shape. The core output steps are executable after a prediction step has published `predictions`; task-specific preparation and prediction are supplied by your application or plugin.

## 1. Install the runtime and required components

```bash
python -m pip install chimera-ml
# Install the task plugin that owns your prediction step.
python -m pip install -e path/to/your_plugin
chimera-ml doctor
chimera-ml plugins list
chimera-ml registry list --type inference_steps
```

The plugin must be installed in the same environment as the CLI. A registry key visible in a different Python environment is not available to this command.

## 2. Prepare the input and checkpoint

The CLI requires an existing input file:

```bash
test -f input.mp4
```

Obtain a checkpoint compatible with the plugin prediction step. A checkpoint may be a local file or an HTTP(S) reference resolved by `resolve_checkpoints_step`. Checkpoint resolution does not load weights into a model; the downstream prediction step must read the named path and load a compatible state dictionary.

## 3. Write an inference YAML

```yaml
pipeline:
  name: application-demo
  parallel: false
runtime:
  device: auto
steps:
  - name: resolve_checkpoints_step
    params:
      cache_dir: model_cache
      checkpoints:
        fusion: /absolute/path/to/fusion.pt
  - name: your_prediction_step
    params:
      checkpoint_key: fusion
  - name: write_json_predictions_step
    params:
      output_path: predictions.json
```

`your_prediction_step` is intentionally a placeholder: verify the actual key and parameters with your plugin documentation and `registry list`. Use [Custom prediction steps](custom-steps.md) if you own the component.

## 4. Validate the right configuration

`chimera-ml validate-config` constructs and validates `ExperimentConfig`, which is the training/evaluation configuration. It is not an inference-YAML validator. For inference, check the YAML shape against [Pipeline configuration](pipeline.md), confirm every `name` with `chimera-ml registry list --type inference_steps`, and let `chimera-ml inference` construct the `InferenceConfig`.

## 5. Run and inspect output

```bash
chimera-ml inference   --input input.mp4   --config-path inference.yaml   --output predictions.json   --device auto   --work-dir runs/video-001
```

The supported options are:

- `--input/-i`: required input file path. The CLI checks that it exists.
- `--config-path/-c`: required inference YAML path.
- `--output/-o`: optional JSON path. In sequential mode it creates or overrides `write_json_predictions_step`; in parallel mode that step must already be declared with dependencies.
- `--device`: optional `auto`, `cpu`, or `cuda`; the default comes from `runtime.device`, whose default is `auto`.
- `--work-dir`: optional working directory for caches and intermediates. Without it, the CLI creates a temporary `chimera-inference-*` directory.

The JSON writer produces:

```json
{
  "input": "input.mp4",
  "predictions": [{"label": "example", "score": 0.8}]
}
```

The exact prediction object is owned by the plugin step. The core writer does not add confidence calibration, labels, or task metadata. `print_json_predictions_step` prints the same shape without writing a file.

## Temporary versus persistent work directories

Use a persistent `--work-dir` when checkpoint downloads or intermediate features should be reused and inspected. Use the default temporary directory for disposable runs; the CLI creates it, but application steps own the artifacts placed there. A relative checkpoint cache path is resolved relative to `work_dir`.
