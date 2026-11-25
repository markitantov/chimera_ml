# Built-in inference steps

The core package currently registers four public inference steps. Confirm the live environment with:

```bash
chimera-ml registry list --type inference_steps
```

Task-specific extraction, feature, fusion, and prediction steps in example projects are plugin components, not core built-ins.

## `resolve_checkpoints_step`

**Purpose:** resolve named local or HTTP(S) checkpoint references and publish paths for downstream steps.

| Parameter | Default | Requirement/behavior |
| --- | --- | --- |
| `checkpoints` | required | Mapping from logical key to local path or HTTP(S) URL. |
| `cache_dir` | `"checkpoints"` | Relative to `ctx.work_dir` unless absolute. |
| `force_download` | `false` | Redownload an existing remote cache file when true. |
| `chunk_size` | `8192` | HTTP streaming chunk size in bytes. |

```yaml
- name: resolve_checkpoints_step
  params:
    cache_dir: model_cache
    checkpoints:
      fusion: https://example.org/models/fusion.pt
```

The step merges with an existing `artifacts["checkpoints"]` mapping and writes the resolved absolute paths to that key. It also writes the resolved cache directory to `artifacts["cache_dir"]`. Local paths must be files; remote downloads use a temporary `.part` file and remove it on failure. This step does not load a state dictionary or construct a model.

## `write_json_predictions_step`

**Purpose:** write the conventional predictions artifact and input path as JSON.

| Parameter | Default | Behavior |
| --- | --- | --- |
| `output_path` | `null` | Explicit output file; when omitted, replace the input suffix with `.json`. |

```yaml
- name: write_json_predictions_step
  params:
    output_path: outputs/predictions.json
```

It requires `ctx.predictions` and publishes the written `Path` as `artifacts["output_path"]`. The parent directory is created. If predictions are missing, it raises `ValueError`.

## `print_json_predictions_step`

**Purpose:** print the same `{input, predictions}` payload without creating a file.

```yaml
- name: print_json_predictions_step
```

It requires `predictions`; otherwise it raises `ValueError`. It has no parameters and publishes no output path.

## `xai_step`

**Purpose:** wrap a registered callback-capable prediction step and publish modality intervention results as `artifacts["xai"]`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `step` | required | Nested step mapping or already-built step. |
| `ablation_policy` | `zero` | `zero`, `mask`, or `remove`. |
| `modalities` | `null` | Iterable of modality names; null uses modalities in each batch. |
| `score_fn` | `null` | Python callable selecting a batch-aligned score; not an arbitrary YAML function. |
| `diagnostic_keys` | `null` | Optional auxiliary output names to copy. |
| `output_mask_key` | `null` | Mask key or Python callable for valid output positions. |
| `validity_fn` | `null` | Optional Python callable rejecting an ablated sample. |
| `temporal_axis` | `null` | Positive batch-relative output axis for summary localization. |

```yaml
- name: xai_step
  params:
    step:
      name: your_prediction_step
      params: {}
    modalities: [audio, video]
    ablation_policy: zero
    diagnostic_keys: [features]
    temporal_axis: 1
```

The nested step must expose `on_prediction`; otherwise the wrapper raises `TypeError`. See [Explainable AI](../user-guide/xai.md) for the full artifact and interpretation contract.
