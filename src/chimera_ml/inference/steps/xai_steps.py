"""Universal XAI wrapper and modality intervention engine."""

import json
import math
import numbers
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import InterventionValidation, ModelOutput
from chimera_ml.inference.context import InferenceContext


def _batch_size(batch: Batch) -> int:
    for value in batch.inputs.values():
        return int(value.shape[0])

    masks = batch.get_masks() or {}
    for value in masks.values():
        value = torch.as_tensor(value)
        if value.ndim:
            return int(value.shape[0])

    raise ValueError("Cannot determine batch size from an empty batch.")


def _availability(batch: Batch, modality: str, batch_size: int) -> torch.Tensor:
    device = next(iter(batch.inputs.values())).device if batch.inputs else torch.device("cpu")
    if modality not in batch.inputs:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    value = batch.inputs[modality]
    temporal_width = value.shape[1] if value.ndim >= 3 else 1
    dedicated = batch.get_masks(f"{modality}_temporal_mask")
    if dedicated is not None:
        mask = torch.as_tensor(dedicated, device=device).bool()
        if mask.shape != (batch_size, temporal_width):
            raise ValueError(
                f"Mask '{modality}_temporal_mask' must have shape "
                f"({batch_size}, {temporal_width}), got {tuple(mask.shape)}."
            )

        return mask.any(dim=1)

    presence = batch.get_masks(f"{modality}_mask")
    if presence is None:
        resolved = torch.ones((batch_size, temporal_width), dtype=torch.bool, device=device)
    else:
        resolved = torch.as_tensor(presence, device=device).bool()
        if resolved.ndim == 0 or resolved.shape[0] != batch_size:
            raise ValueError(
                f"Mask '{modality}_mask' must have batch dimension {batch_size}, got {tuple(resolved.shape)}."
            )

        if resolved.ndim == 1:
            resolved = resolved[:, None].expand(-1, temporal_width)
        elif resolved.ndim != 2:
            raise ValueError(f"Mask '{modality}_mask' must have shape (B,) or (B, T), got {tuple(resolved.shape)}.")
        elif resolved.shape[1] != temporal_width:
            raise ValueError(
                f"Mask '{modality}_mask' must have temporal width {temporal_width}, got {resolved.shape[1]}."
            )

    shared = batch.get_masks("sequence_mask")
    if shared is not None:
        shared = torch.as_tensor(shared, device=device).bool()
        if shared.ndim == 0 or shared.shape[0] != batch_size:
            raise ValueError(f"Mask 'sequence_mask' must have batch dimension {batch_size}, got {tuple(shared.shape)}.")

        if shared.ndim == 1:
            shared = shared[:, None]
        elif shared.ndim != 2:
            raise ValueError(f"Mask 'sequence_mask' must have shape (B,) or (B, T), got {tuple(shared.shape)}.")

        if shared.shape == resolved.shape:
            resolved = resolved & shared

    return resolved.flatten(1).any(dim=1)


def _drop_modality_masks(masks: dict[str, Any], modality: str) -> None:
    for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
        masks.pop(key, None)

    nested = masks.get("mask")
    if isinstance(nested, Mapping):
        nested = dict(nested)
        for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
            nested.pop(key, None)

        masks["mask"] = nested


def _disable_modality_masks(masks: dict[str, Any], modality: str) -> bool:
    changed = False
    for key in (f"{modality}_mask", f"{modality}_temporal_mask"):
        if key in masks:
            masks[key] = torch.zeros_like(torch.as_tensor(masks[key]).bool())
            changed = True

    nested = masks.get("mask")
    if isinstance(nested, Mapping):
        nested = dict(nested)
        for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
            if key in nested:
                nested[key] = torch.zeros_like(torch.as_tensor(nested[key]).bool())
                changed = True

        masks["mask"] = nested

    return changed


def _intervene(batch: Batch, modality: str, policy: str) -> Batch:
    if modality not in batch.inputs:
        raise ValueError(f"Modality '{modality}' is not present in the batch.")

    inputs = dict(batch.inputs)
    has_canonical_masks = isinstance(batch.masks, dict)
    masks = deepcopy(batch.masks) if has_canonical_masks else deepcopy(batch.get_masks() or {})
    meta = deepcopy(batch.meta)
    value = batch.inputs[modality]

    if policy == "zero":
        inputs[modality] = torch.zeros_like(value)
    elif policy == "remove":
        del inputs[modality]
        _drop_modality_masks(masks, modality)
    elif policy == "mask":
        changed = _disable_modality_masks(masks, modality)
        if not changed:
            masks[f"{modality}_mask"] = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)
    else:
        raise ValueError("ablation_policy must be zero, mask, or remove.")

    if policy == "remove" and isinstance(meta, dict) and isinstance(meta.get("masks"), Mapping):
        legacy = dict(meta["masks"])
        _drop_modality_masks(legacy, modality)
        meta["masks"] = legacy

    return Batch(inputs, batch.targets, masks if has_canonical_masks or masks else None, meta)


def _select_value(value: Any, indices: Tensor, batch_size: int) -> Any:
    if torch.is_tensor(value):
        return value.index_select(0, indices) if value.ndim and value.shape[0] == batch_size else value
    if isinstance(value, Mapping):
        return {key: _select_value(item, indices, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        return [value[int(index)] for index in indices.detach().cpu().tolist()]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[int(index)] for index in indices.detach().cpu().tolist())
    return value


def _select_batch(batch: Batch, indices: Tensor, batch_size: int) -> Batch:
    return Batch(
        _select_value(batch.inputs, indices, batch_size),
        _select_value(batch.targets, indices, batch_size),
        _select_value(batch.masks, indices, batch_size),
        _select_value(batch.meta, indices, batch_size),
    )


def _expand_value(value: Any, indices: Tensor, batch_size: int) -> Any:
    if torch.is_tensor(value):
        if not value.ndim or value.shape[0] != indices.numel():
            return value
        shape = (batch_size, *value.shape[1:])
        expanded = value.new_zeros(shape)
        return expanded.index_copy(0, indices, value)
    if isinstance(value, Mapping):
        return {key: _expand_value(item, indices, batch_size) for key, item in value.items()}
    return value


def _expand_output(output: ModelOutput, indices: Tensor, batch_size: int) -> ModelOutput:
    return ModelOutput(
        preds=_expand_value(output.preds, indices, batch_size),
        aux=_expand_value(output.aux, indices, batch_size) if output.aux is not None else None,
    )


def _prevalidate_intervention(
    model: nn.Module, original: Batch, intervened: Batch, modality: str, policy: str, present: Tensor
) -> tuple[Tensor, str | None]:
    validator = getattr(model, "validate_intervention", None)
    if not callable(validator):
        return present, None
    result = validator(original, intervened, modality, policy)
    if not isinstance(result, InterventionValidation):
        raise TypeError("validate_intervention must return InterventionValidation.")
    supported = torch.as_tensor(result.supported, device=present.device).bool()
    if supported.ndim == 0:
        supported = supported.expand_as(present)
    elif supported.shape != present.shape:
        raise ValueError("validate_intervention supported mask must have shape (batch_size,).")
    return present & supported, result.reason


def _json(value: Any) -> Any:
    if torch.is_tensor(value):
        return _json(value.detach().cpu().tolist())

    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]

    return value


def _sample(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value

        return value[index]

    if isinstance(value, Mapping):
        return {key: _sample(item, index) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return value[index]

    return value


def _validate_batch_aligned(value: Any, batch_size: int, name: str) -> None:
    if torch.is_tensor(value):
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(f"Diagnostic '{name}' must be batch-aligned with size {batch_size}.")

        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_batch_aligned(item, batch_size, f"{name}.{key}")

        return
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"Diagnostic '{name}' must be batch-aligned with size {batch_size}.")

        return

    raise ValueError(f"Diagnostic '{name}' must be a batch-aligned tensor, mapping, or sequence.")


def _masked_json(value: Any, valid: Any) -> Any:
    value = _json(value)
    valid = _json(valid)
    if isinstance(valid, list):
        if not isinstance(value, list) or len(value) != len(valid):
            raise ValueError("Output validity shape does not match score shape.")

        return [
            None if not item_valid else _masked_json(item, item_valid)
            for item, item_valid in zip(value, valid, strict=True)
        ]

    return value if bool(valid) else None


def _score(output: ModelOutput, batch_size: int, score_fn: Callable[[ModelOutput], Tensor] | None) -> Tensor:
    score = output.preds if score_fn is None else score_fn(output)
    if not torch.is_tensor(score):
        raise TypeError("score_fn must return a torch.Tensor.")

    if score.ndim == 0 or score.shape[0] != batch_size:
        raise ValueError(f"Scores must be batch-aligned with size {batch_size}, got {tuple(score.shape)}.")

    return score


def _validity(
    batch: Batch,
    output: ModelOutput,
    score: Tensor,
    output_mask_key: str | Callable[[Batch, ModelOutput], Tensor] | None,
) -> Tensor:
    if output_mask_key is None:
        return torch.ones_like(score, dtype=torch.bool)

    value = output_mask_key(batch, output) if callable(output_mask_key) else batch.get_masks(output_mask_key)
    if value is None:
        raise ValueError(f"Output validity mask '{output_mask_key}' is missing.")

    mask = torch.as_tensor(value, device=score.device).bool()
    if mask.ndim == 0 or mask.shape[0] != score.shape[0]:
        raise ValueError(f"Output validity mask must have batch dimension {score.shape[0]}, got {tuple(mask.shape)}.")

    while mask.ndim < score.ndim:
        mask = mask.unsqueeze(-1)

    try:
        return mask.expand_as(score)
    except RuntimeError as error:
        raise ValueError(
            f"Output validity mask shape {tuple(mask.shape)} does not match score {tuple(score.shape)}."
        ) from error


def _sample_ids(batch: Batch, batch_index: int | None) -> list[str]:
    size = _batch_size(batch)
    raw = (batch.meta or {}).get("sample_ids")
    if raw is None:
        prefix = "sample" if batch_index is None else f"batch-{batch_index}"
        return [f"{prefix}-{index}" for index in range(size)]

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Iterable):
        raise ValueError("batch.meta['sample_ids'] must be an iterable with one value per sample.")

    values = list(raw)
    if len(values) != size:
        raise ValueError(f"batch.meta['sample_ids'] has {len(values)} values, expected {size}.")

    return [str(value) for value in values]


def _missing(reason: str) -> dict[str, Any]:
    return {"status": "already_missing", "reason": reason, "ablated": None, "delta": None, "absolute_delta": None}


def _unsupported(reason: str) -> dict[str, Any]:
    return {"status": "unsupported", "reason": reason, "ablated": None, "delta": None, "absolute_delta": None}


def _require_eval(model: nn.Module) -> None:
    if any(module.training for module in model.modules()):
        raise ValueError("model and all nested modules must already be in evaluation mode.")


def _sample_metadata(batch: Batch, index: int, batch_size: int) -> dict[str, Any]:
    meta = batch.meta or {}
    result: dict[str, Any] = {}
    sample_meta = meta.get("sample_meta")
    if isinstance(sample_meta, list) and len(sample_meta) == batch_size and isinstance(sample_meta[index], Mapping):
        result.update(_json(sample_meta[index]))

    for key in ("window_id", "start", "end", "timestamp", "duration", "start_time", "end_time"):
        if key not in meta:
            continue

        value = meta[key]
        if torch.is_tensor(value):
            if value.ndim and value.shape[0] == batch_size:
                result[key] = _json(_sample(value, index))
        elif isinstance(value, (list, tuple)) and len(value) == batch_size:
            result[key] = _json(_sample(value, index))
        else:
            result[key] = _json(value)

    return result


def explain_batch(
    model: torch.nn.Module,
    batch: Batch,
    *,
    baseline_output: ModelOutput | None = None,
    score_fn: Callable[[ModelOutput], Tensor] | None = None,
    modalities: Iterable[str] | None = None,
    ablation_policy: str = "zero",
    diagnostic_keys: Iterable[str] | None = None,
    output_mask_key: str | Callable[[Batch, ModelOutput], Tensor] | None = None,
    validity_fn: Callable[[str, Batch, ModelOutput], bool] | None = None,
    batch_index: int | None = None,
) -> dict[str, Any]:
    """Compare a baseline ModelOutput with one intervention per modality."""
    if ablation_policy not in {"zero", "mask", "remove"}:
        raise ValueError("ablation_policy must be zero, mask, or remove.")

    _require_eval(model)
    size = _batch_size(batch)
    names = tuple(modalities if modalities is not None else batch.inputs)
    with torch.inference_mode():
        baseline = baseline_output if baseline_output is not None else model(batch)
        original = _score(baseline, size, score_fn)
        baseline_valid = _validity(batch, baseline, original, output_mask_key)
        diagnostics: dict[str, Any] = {}
        for key in sorted(diagnostic_keys or ()):
            if not baseline.aux or key not in baseline.aux:
                continue

            _validate_batch_aligned(baseline.aux[key], size, key)
            diagnostics[key] = baseline.aux[key]

        availability = {name: _availability(batch, name, size) for name in names}
        results: dict[str, list[dict[str, Any]]] = {}
        for name in names:
            present = availability[name]
            if not bool(present.any()):
                results[name] = [_missing("modality is unavailable for this sample") for _ in range(size)]
                continue
            try:
                intervened = _intervene(batch, name, ablation_policy)
            except ValueError as error:
                results[name] = [_unsupported(str(error)) for _ in range(size)]
                continue

            supported, validation_reason = _prevalidate_intervention(
                model, batch, intervened, name, ablation_policy, present
            )
            if not bool(supported.any()):
                reason = validation_reason or "intervention is unsupported"
                results[name] = [
                    _missing("modality is unavailable for this sample")
                    if not bool(present[index])
                    else _unsupported(reason)
                    for index in range(size)
                ]
                continue

            supported_indices = supported.nonzero(as_tuple=False).flatten()
            forward_batch = _select_batch(intervened, supported_indices, size)
            output = model(forward_batch)
            output = _expand_output(output, supported_indices, size)
            changed = _score(output, size, score_fn)
            if changed.shape != original.shape:
                raise ValueError(
                    f"Baseline and ablated scores differ: {tuple(original.shape)} vs {tuple(changed.shape)}."
                )

            changed_valid = _validity(intervened, output, changed, output_mask_key)
            common_valid = baseline_valid & changed_valid
            common_valid &= supported.view((size,) + (1,) * (common_valid.ndim - 1))
            rejected = ~supported.clone().to(device=common_valid.device)
            if validity_fn is not None:
                valid = validity_fn(name, intervened, output)
                if torch.is_tensor(valid):
                    if valid.ndim == 0 or valid.shape[0] != size:
                        raise ValueError("validity_fn result must be batch-aligned.")

                    valid = valid.to(common_valid.device).bool()
                    rejected = ~valid if valid.ndim == 1 else ~valid.flatten(1).any(dim=1)
                    while valid.ndim < common_valid.ndim:
                        valid = valid.unsqueeze(-1)

                    valid = valid.expand_as(common_valid)
                    common_valid &= valid
                elif not valid:
                    rejected.fill_(True)
                    common_valid.zero_()

            delta = original - changed
            results[name] = []
            for index in range(size):
                if not bool(present[index]):
                    results[name].append(_missing("modality is unavailable for this sample"))
                    continue

                if bool(rejected[index]):
                    results[name].append(_unsupported("validity_fn rejected the ablated sample"))
                    continue

                valid = common_valid[index]
                if not bool(valid.any()):
                    results[name].append(_unsupported("no common valid output positions remain"))
                    continue

                results[name].append(
                    {
                        "status": "evaluated",
                        "ablated": _masked_json(_sample(changed, index), valid),
                        "delta": _masked_json(_sample(delta, index), valid),
                        "absolute_delta": _masked_json(_sample(delta.abs(), index), valid),
                    }
                )

        sample_ids = _sample_ids(batch, batch_index)
        samples = [
            {
                "sample_id": sample_ids[index],
                "baseline": _masked_json(_sample(original, index), _sample(baseline_valid, index)),
                "modalities": {name: results[name][index] for name in names},
                "diagnostics": {key: _json(_sample(value, index)) for key, value in diagnostics.items()},
                "metadata": _sample_metadata(batch, index, size),
            }
            for index in range(size)
        ]
        result = {
            "schema_version": 1,
            "method": "leave_one_modality_out",
            "ablation_policy": ablation_policy,
            "modalities": list(names),
            "samples": samples,
        }
        json.dumps(result, allow_nan=False)
        return result


def _numeric_paths(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], float]:
    if isinstance(value, Mapping):
        result: dict[tuple[Any, ...], float] = {}
        for key, item in value.items():
            result.update(_numeric_paths(item, (*path, str(key))))

        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_numeric_paths(item, (*path, index)))

        return result
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return {path: float(value)}

    return {}


def _rank_details(
    details: dict[str, Any],
) -> tuple[list[str], list[list[str]], str | None, list[str], str]:
    comparable = [
        name
        for name, item in details.items()
        if item["common_comparison_count"] and item["mean_absolute_delta"] is not None
    ]
    ordered = sorted(comparable, key=lambda name: (-details[name]["mean_absolute_delta"], name))
    ties: list[list[str]] = []
    if ordered:
        current = [ordered[0]]
        for name in ordered[1:]:
            if details[name]["mean_absolute_delta"] == details[current[0]]["mean_absolute_delta"]:
                current.append(name)
            else:
                if len(current) > 1:
                    ties.append(current)
                current = [name]
        if len(current) > 1:
            ties.append(current)

    total_abs = sum(details[name]["mean_absolute_delta"] for name in ordered)
    if total_abs > 0:
        for index, name in enumerate(ordered):
            previous = ordered[index - 1] if index else None
            details[name]["rank"] = (
                details[previous]["rank"]
                if previous and details[previous]["mean_absolute_delta"] == details[name]["mean_absolute_delta"]
                else index + 1
            )
            details[name]["relative_influence"] = details[name]["mean_absolute_delta"] / total_abs
        dominant_modalities = [
            name
            for name in ordered
            if details[name]["mean_absolute_delta"] == details[ordered[0]]["mean_absolute_delta"]
        ]
        dominant = dominant_modalities[0] if len(dominant_modalities) == 1 else None
        status = "evaluated"
    elif ordered:
        dominant_modalities = []
        dominant = None
        status = "no_measurable_influence"
    else:
        dominant_modalities = []
        dominant = None
        status = (
            "no_common_comparison" if any(item["status"] == "evaluated" for item in details.values()) else "unavailable"
        )

    return ordered, ties, dominant, dominant_modalities, status


def _comparison_summary(
    paths_by_modality: Mapping[str, dict[tuple[Any, ...], float]],
    statuses: Mapping[str, str],
) -> dict[str, Any]:
    modalities = list(statuses)
    evaluated = {name: paths_by_modality.get(name, {}) for name in modalities if statuses[name] == "evaluated"}
    common: set[tuple[Any, ...]] = set()
    if evaluated:
        common = set.intersection(*(set(paths) for paths in evaluated.values()))

    details: dict[str, Any] = {}
    for name in modalities:
        status = statuses[name]
        paths = paths_by_modality.get(name, {})
        item: dict[str, Any] = {
            "status": status,
            "mean_signed_delta": None,
            "mean_absolute_delta": None,
            "relative_influence": None,
            "rank": None,
            "valid_comparison_count": len(paths),
            "common_comparison_count": 0,
        }

        if status == "evaluated" and not paths:
            item["status"] = "insufficient_data"

        if status == "evaluated" and common:
            values = [paths[position] for position in common if position in paths]
            item["common_comparison_count"] = len(values)
            item["mean_signed_delta"] = sum(values) / len(values)
            item["mean_absolute_delta"] = sum(abs(value) for value in values) / len(values)
        details[name] = item

    ordered, ties, dominant, dominant_modalities, status = _rank_details(details)

    reason = None
    if status == "unavailable":
        reason = "no evaluated modality comparisons"
    elif status == "no_common_comparison":
        reason = "no common valid output positions remain"
    elif status == "no_measurable_influence":
        reason = "all comparable influence magnitudes are zero"

    return {
        "status": status,
        "reason": reason,
        "ranking": ordered,
        "ties": ties,
        "dominant_modality": dominant,
        "dominant_modalities": dominant_modalities,
        "comparable_modalities": ordered,
        "common_comparison_count": len(common),
        "modalities": details,
    }


def _temporal_paths(
    paths_by_modality: Mapping[str, dict[tuple[Any, ...], float]],
    temporal_axis: int,
) -> dict[int, dict[str, dict[tuple[Any, ...], float]]]:
    position_index = temporal_axis - 1
    positions: dict[int, dict[str, dict[tuple[Any, ...], float]]] = {}
    for name, paths in paths_by_modality.items():
        for path, value in paths.items():
            if len(path) <= position_index or not isinstance(path[position_index], int):
                continue

            position = path[position_index]
            positions.setdefault(position, {}).setdefault(name, {})[path] = value

    return positions


def _has_score_axis(value: Any, axis: int, depth: int = 0) -> bool:
    if isinstance(value, list):
        if depth == axis:
            return True

        return any(_has_score_axis(item, axis, depth + 1) for item in value)
    if isinstance(value, Mapping):
        return any(_has_score_axis(item, axis, depth) for item in value.values())

    return False


def _sample_summary(sample: Mapping[str, Any], modalities: list[str], temporal_axis: int | None) -> dict[str, Any]:
    entries = sample.get("modalities") or {}
    statuses = {name: str((entries.get(name) or {}).get("status", "already_missing")) for name in modalities}
    paths = {name: _numeric_paths((entries.get(name) or {}).get("delta")) for name in modalities}
    result = _comparison_summary(paths, statuses)
    result["sample_id"] = sample.get("sample_id")
    result["sample_coverage"] = 1.0 if result["common_comparison_count"] else 0.0

    if temporal_axis is not None:
        temporal = _temporal_paths(paths, temporal_axis)
        deltas = [(entries.get(name) or {}).get("delta") for name in modalities if entries.get(name)]
        if any(delta is not None for delta in deltas) and not all(
            _has_score_axis(delta, temporal_axis - 1) for delta in deltas if delta is not None
        ):
            raise ValueError("temporal_axis does not identify a list-like score dimension.")

        result["temporal"] = [
            {
                "position": position,
                **_comparison_summary(group_paths, statuses),
            }
            for position, group_paths in sorted(temporal.items())
        ]
    return result


def summarize_explanations(
    explanations: Mapping[str, Any],
    *,
    temporal_axis: int | None = None,
) -> dict[str, Any]:
    """Summarize JSON-compatible modality explanations without model execution."""
    if temporal_axis is not None and (not isinstance(temporal_axis, int) or temporal_axis < 1):
        raise ValueError("temporal_axis must be a positive batch-relative axis or None.")

    samples = explanations.get("samples")
    if not isinstance(samples, list):
        raise ValueError("explanations['samples'] must be a list.")

    modalities = list(explanations.get("modalities") or [])
    for sample in samples:
        for name in sample.get("modalities") or {}:
            if name not in modalities:
                modalities.append(name)

    sample_summaries = [_sample_summary(sample, modalities, temporal_axis) for sample in samples]
    paths_by_sample = []
    for sample in samples:
        entries = sample.get("modalities") or {}
        paths_by_sample.append({name: _numeric_paths((entries.get(name) or {}).get("delta")) for name in modalities})

    statuses_by_sample = [
        {
            name: str((sample.get("modalities") or {}).get(name, {}).get("status", "already_missing"))
            for name in modalities
        }
        for sample in samples
    ]

    availability = {
        name: {
            "availability_count": sum(status != "already_missing" for status in statuses),
            "evaluated_sample_count": sum(status == "evaluated" for status in statuses),
            "missing_sample_count": sum(status == "already_missing" for status in statuses),
            "unsupported_sample_count": sum(status == "unsupported" for status in statuses),
        }
        for name, statuses in ((name, [row[name] for row in statuses_by_sample]) for name in modalities)
    }

    candidates = [
        name
        for name in modalities
        if any(
            statuses[name] == "evaluated" and paths_by_sample[index][name]
            for index, statuses in enumerate(statuses_by_sample)
        )
    ]

    if len(candidates) > 1:
        joint_indices = [
            index
            for index, statuses in enumerate(statuses_by_sample)
            if all(statuses[name] == "evaluated" and paths_by_sample[index][name] for name in candidates)
        ]
    else:
        joint_indices = [
            index
            for index, statuses in enumerate(statuses_by_sample)
            if candidates and statuses[candidates[0]] == "evaluated" and paths_by_sample[index][candidates[0]]
        ]

    global_paths: dict[str, dict[tuple[Any, ...], float]] = {name: {} for name in modalities}
    sample_contributions: dict[str, list[tuple[float, float]]] = {name: [] for name in candidates}
    valid_counts = dict.fromkeys(modalities, 0)
    common_counts = dict.fromkeys(modalities, 0)
    for index, statuses in enumerate(statuses_by_sample):
        for name in modalities:
            if statuses[name] == "evaluated":
                valid_counts[name] += len(paths_by_sample[index][name])

    common_sample_count = 0
    for index in joint_indices:
        selected = {name: paths_by_sample[index][name] for name in candidates}
        common = set.intersection(*(set(paths) for paths in selected.values()))
        if not common:
            continue

        common_sample_count += 1
        for name in candidates:
            common_counts[name] += len(common)
            values = [selected[name][position] for position in common]
            sample_contributions[name].append(
                (sum(values) / len(values), sum(abs(value) for value in values) / len(values))
            )
            global_paths[name].update(
                {(index, *position): value for position, value in selected[name].items() if position in common}
            )

    global_statuses = {
        name: "evaluated"
        if name in candidates
        else ("unsupported" if availability[name]["unsupported_sample_count"] else "already_missing")
        for name in modalities
    }
    global_result = _comparison_summary(global_paths, global_statuses)
    for name in candidates:
        contributions = sample_contributions[name]
        if contributions:
            item = global_result["modalities"][name]
            item["mean_signed_delta"] = sum(value[0] for value in contributions) / len(contributions)
            item["mean_absolute_delta"] = sum(value[1] for value in contributions) / len(contributions)

    for name in modalities:
        global_result["modalities"][name]["valid_comparison_count"] = valid_counts[name]
        global_result["modalities"][name]["common_comparison_count"] = common_counts[name]
    (
        global_result["ranking"],
        global_result["ties"],
        global_result["dominant_modality"],
        global_result["dominant_modalities"],
        global_result["status"],
    ) = _rank_details(global_result["modalities"])

    for name in modalities:
        availability[name].update(global_result["modalities"][name])
        availability[name]["comparable_sample_count"] = common_sample_count if name in candidates else 0
        availability[name]["sample_coverage"] = (
            availability[name]["comparable_sample_count"] / len(samples) if samples else 0.0
        )

    if len(candidates) > 1 and common_sample_count == 0:
        global_result["status"] = "insufficient_common_support"
        global_result["reason"] = "modalities have no jointly comparable samples"
        global_result["ranking"] = []
        global_result["ties"] = []
        global_result["dominant_modality"] = None
        global_result["dominant_modalities"] = []
        for item in global_result["modalities"].values():
            item["mean_signed_delta"] = None
            item["mean_absolute_delta"] = None
            item["relative_influence"] = None
            item["rank"] = None

    global_result["total_samples"] = len(samples)
    global_result["comparable_samples"] = common_sample_count
    global_result["sample_coverage"] = common_sample_count / len(samples) if samples else 0.0
    global_result["modalities"] = availability
    global_result["samples"] = sample_summaries
    global_result["temporal_axis"] = temporal_axis

    if temporal_axis is not None:
        temporal_groups: dict[int, dict[str, dict[tuple[Any, ...], float]]] = {}
        for sample_index, paths in enumerate(paths_by_sample):
            for position, group in _temporal_paths(paths, temporal_axis).items():
                for name, values in group.items():
                    temporal_groups.setdefault(position, {}).setdefault(name, {}).update(
                        {(sample_index, *path): value for path, value in values.items()}
                    )

        global_result["temporal"] = [
            {
                "position": position,
                **_comparison_summary(group_paths, global_statuses),
            }
            for position, group_paths in sorted(temporal_groups.items())
        ]

    json.dumps(global_result, allow_nan=False)
    return global_result


@dataclass
class XAIStep:
    """Run a callback-capable prediction step and collect explanations."""

    step: Any
    ablation_policy: str = "zero"
    modalities: Iterable[str] | None = None
    score_fn: Callable[[ModelOutput], Tensor] | None = None
    diagnostic_keys: Iterable[str] | None = None
    output_mask_key: str | Callable[[Batch, ModelOutput], Tensor] | None = None
    validity_fn: Callable[[str, Batch, ModelOutput], bool] | None = None
    temporal_axis: int | None = None
    _results: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        if not hasattr(self.step, "on_prediction"):
            raise TypeError(
                f"Wrapped step '{type(self.step).__name__}' must expose an 'on_prediction' callback attribute."
            )

        previous = self.step.on_prediction
        if previous is not None and not callable(previous):
            raise TypeError("Wrapped step 'on_prediction' must be callable or None.")

        self._results = []

        def collect(model: nn.Module, batch: Batch, output: ModelOutput) -> None:
            if previous is not None:
                previous(model, batch, output)

            self._results.append(
                explain_batch(
                    model,
                    batch,
                    baseline_output=output,
                    score_fn=self.score_fn,
                    modalities=self.modalities,
                    ablation_policy=self.ablation_policy,
                    diagnostic_keys=self.diagnostic_keys,
                    output_mask_key=self.output_mask_key,
                    validity_fn=self.validity_fn,
                    batch_index=len(self._results),
                )
            )

        self.step.on_prediction = collect
        try:
            result = self.step.run(ctx)
            result.set_artifact("xai", self._combine())
            return result
        finally:
            self.step.on_prediction = previous

    def _combine(self) -> dict[str, Any]:
        modalities: list[str] = []
        samples: list[dict[str, Any]] = []
        for item in self._results:
            for name in item["modalities"]:
                if name not in modalities:
                    modalities.append(name)

            samples.extend(item["samples"])
        for sample in samples:
            sample["modalities"] = {
                name: sample["modalities"].get(name, _missing("modality was not present in this batch"))
                for name in modalities
            }

        result = {
            "schema_version": 2,
            "method": "leave_one_modality_out",
            "ablation_policy": self.ablation_policy,
            "modalities": modalities,
            "samples": samples,
            "batches": len(self._results),
        }
        result["summary"] = summarize_explanations(result, temporal_axis=self.temporal_axis)
        json.dumps(result, allow_nan=False)
        return result


@INFERENCE_STEPS.register("xai_step")
def xai_step(step: Mapping[str, Any] | Any, **params: Any) -> XAIStep:
    """Build XAIStep around another registered inference step."""
    if isinstance(step, Mapping):
        from chimera_ml.inference.builders import build_inference_step

        step = build_inference_step(dict(step))

    return XAIStep(step=step, **params)
