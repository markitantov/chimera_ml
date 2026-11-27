"""Minimal leave-one-modality-out prediction sensitivity analysis."""

import json
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
    return next(iter(batch.inputs.values())).shape[0]


def _availability(batch: Batch, modality: str, size: int) -> Tensor:
    if modality not in batch.inputs:
        return torch.zeros(size, dtype=torch.bool)
    value = batch.inputs[modality]
    width = value.shape[1] if value.ndim > 2 else 1
    temporal = batch.get_masks(f"{modality}_temporal_mask")
    if temporal is not None:
        return torch.as_tensor(temporal, device=value.device).bool().reshape(size, width).any(1)
    mask = batch.get_masks(f"{modality}_mask")
    if mask is None:
        present = torch.ones((size, width), dtype=torch.bool, device=value.device)
    else:
        present = torch.as_tensor(mask, device=value.device).bool()
        present = present[:, None].expand(-1, width) if present.ndim == 1 else present
    sequence = batch.get_masks("sequence_mask")
    if sequence is not None:
        sequence = torch.as_tensor(sequence, device=value.device).bool()
        sequence = sequence[:, None] if sequence.ndim == 1 else sequence
        if sequence.shape == present.shape:
            present &= sequence
    return present.any(1)


def _remove_masks(masks: dict[str, Any], modality: str) -> None:
    for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
        masks.pop(key, None)
    nested = masks.get("mask")
    if isinstance(nested, dict):
        nested = dict(nested)
        for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
            nested.pop(key, None)
        masks["mask"] = nested


def _intervene(batch: Batch, modality: str, policy: str) -> Batch:
    if modality not in batch.inputs:
        raise ValueError(f"Modality '{modality}' is not present in the batch.")
    inputs = dict(batch.inputs)
    canonical = isinstance(batch.masks, dict)
    masks = deepcopy(batch.masks) if canonical else deepcopy(batch.get_masks() or {})
    value = inputs[modality]
    if policy == "zero":
        inputs[modality] = torch.zeros_like(value)
    elif policy == "remove":
        del inputs[modality]
        _remove_masks(masks, modality)
    elif policy == "mask":
        changed = False
        for key in (f"{modality}_mask", f"{modality}_temporal_mask"):
            if key in masks:
                masks[key] = torch.zeros_like(torch.as_tensor(masks[key]).bool())
                changed = True
        nested = masks.get("mask")
        if isinstance(nested, dict):
            nested = dict(nested)
            for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
                if key in nested:
                    nested[key] = torch.zeros_like(torch.as_tensor(nested[key]).bool())
                    changed = True
            masks["mask"] = nested
        if not changed:
            masks[f"{modality}_mask"] = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)
    else:
        raise ValueError("ablation_policy must be zero, mask, or remove.")
    meta = deepcopy(batch.meta)
    if policy == "remove" and isinstance(meta, dict) and isinstance(meta.get("masks"), Mapping):
        _remove_masks(meta["masks"], modality)
    return Batch(inputs, batch.targets, masks if canonical or masks else None, meta)


def _json(value: Any) -> Any:
    return value.detach().cpu().tolist() if torch.is_tensor(value) else value


def _score(output: ModelOutput, size: int, score_fn: Callable[[ModelOutput], Tensor] | None) -> Tensor:
    score = output.preds if score_fn is None else score_fn(output)
    if not torch.is_tensor(score):
        raise TypeError("score_fn must return a torch.Tensor.")
    if score.ndim == 0 or score.shape[0] != size:
        raise ValueError("Scores must be batch-aligned with the input batch.")
    return score


def _validity(
    batch: Batch,
    output: ModelOutput,
    score: Tensor,
    key: str | Callable[[Batch, ModelOutput], Tensor] | None,
) -> Tensor:
    if key is None:
        return torch.ones_like(score, dtype=torch.bool)
    value = key(batch, output) if callable(key) else batch.get_masks(key)
    if value is None:
        raise ValueError(f"Output validity mask '{key}' is missing.")
    mask = torch.as_tensor(value, device=score.device).bool()
    while mask.ndim < score.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(score)


def _supported(
    model: nn.Module, batch: Batch, changed: Batch, modality: str, policy: str, available: Tensor
) -> tuple[Tensor, str | None]:
    validator = getattr(model, "validate_intervention", None)
    if not callable(validator):
        return available, None
    result = validator(batch, changed, modality, policy)
    if isinstance(result, InterventionValidation):
        value, reason = result.supported, result.reason
    else:
        value, reason = result, "intervention is unsupported"
    supported = torch.as_tensor(value, device=available.device).bool()
    if supported.ndim == 0:
        supported = supported.expand_as(available)
    if supported.shape != available.shape:
        raise ValueError("validate_intervention must return a bool or a [B] boolean tensor.")
    return available & supported, reason


def _ids(batch: Batch, batch_index: int | None) -> list[str]:
    raw = (batch.meta or {}).get("sample_ids")
    if raw is not None:
        return [str(value) for value in raw]
    prefix = "sample" if batch_index is None else f"batch-{batch_index}"
    return [f"{prefix}-{index}" for index in range(_batch_size(batch))]


def _entry(status: str, reason: str | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "ablated": None, "delta": None, "absolute_delta": None}


def _masked(value: Tensor, valid: Tensor) -> Any:
    if value.ndim == 0:
        return value.item() if bool(valid) else None
    values = value.detach().cpu().tolist()
    mask = valid.detach().cpu().tolist()
    if isinstance(values, list) and all(isinstance(item, (int, float)) for item in values):
        return [item if keep else None for item, keep in zip(values, mask, strict=True)]
    return values


def explain_batch(
    model: nn.Module,
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
    """Compute ``h(f(x)) - h(f(x without modality))`` for one Batch."""
    if ablation_policy not in {"zero", "mask", "remove"}:
        raise ValueError("ablation_policy must be zero, mask, or remove.")
    if any(module.training for module in model.modules()):
        raise ValueError("model and all nested modules must already be in evaluation mode.")
    size = _batch_size(batch)
    names = list(modalities if modalities is not None else batch.inputs)
    with torch.inference_mode():
        baseline_output = baseline_output if baseline_output is not None else model(batch)
        baseline_score = _score(baseline_output, size, score_fn)
        baseline_valid = _validity(batch, baseline_output, baseline_score, output_mask_key)
        diagnostics = {}
        for key in diagnostic_keys or ():
            if baseline_output.aux and key in baseline_output.aux:
                value = baseline_output.aux[key]
                if value.ndim == 0 or value.shape[0] != size:
                    raise ValueError(f"Diagnostic '{key}' must be batch-aligned with size {size}.")
                diagnostics[key] = value
        result: dict[str, list[dict[str, Any]]] = {}
        for modality in names:
            available = _availability(batch, modality, size)
            entries = [_entry("already_missing", "modality is unavailable for this sample") for _ in range(size)]
            if not available.any():
                result[modality] = entries
                continue
            changed_batch = _intervene(batch, modality, ablation_policy)
            supported, reason = _supported(model, batch, changed_batch, modality, ablation_policy, available)
            if supported.any() and not torch.equal(supported, available):
                supported = torch.zeros_like(supported)
                reason = reason or "mixed-validity intervention cannot be evaluated as one batch"
            if not supported.any():
                for index in available.nonzero().flatten().tolist():
                    entries[index] = _entry("unsupported", reason or "intervention is unsupported")
                result[modality] = entries
                continue
            intervened_output = model(changed_batch)
            intervened_score = _score(intervened_output, size, score_fn)
            if intervened_score.shape != baseline_score.shape:
                raise ValueError("Baseline and intervened scores must have the same shape.")
            valid = baseline_valid & _validity(changed_batch, intervened_output, intervened_score, output_mask_key)
            if validity_fn is not None:
                accepted = validity_fn(modality, changed_batch, intervened_output)
                if torch.is_tensor(accepted):
                    accepted = accepted.bool()
                    if accepted.ndim == 1 and valid.ndim > 1:
                        accepted = accepted.view((accepted.shape[0],) + (1,) * (valid.ndim - 1))
                    valid &= accepted
                elif not accepted:
                    valid.zero_()
            delta = baseline_score - intervened_score
            for index in available.nonzero().flatten().tolist():
                sample_valid = valid[index]
                if not sample_valid.any():
                    reason = (
                        "validity_fn rejected the ablated sample"
                        if validity_fn is not None
                        else "no common valid output positions remain"
                    )
                    entries[index] = _entry("unsupported", reason)
                    continue
                entries[index] = {
                    "status": "evaluated",
                    "reason": None,
                    "ablated": _masked(intervened_score[index], sample_valid),
                    "delta": _masked(delta[index], sample_valid),
                    "absolute_delta": _masked(delta[index].abs(), sample_valid),
                }
            result[modality] = entries
        samples = []
        ids = _ids(batch, batch_index)
        for index, sample_id in enumerate(ids):
            samples.append(
                {
                    "sample_id": sample_id,
                    "baseline": _masked(baseline_score[index], baseline_valid[index]),
                    "modalities": {name: result[name][index] for name in names},
                    "diagnostics": {
                        key: _json(baseline_output.aux[key][index])
                        for key in (diagnostic_keys or ())
                        if baseline_output.aux and key in baseline_output.aux
                    },
                    "metadata": {},
                }
            )
        artifact = {
            "schema_version": 1,
            "method": "leave_one_modality_out",
            "ablation_policy": ablation_policy,
            "modalities": names,
            "samples": samples,
        }
        json.dumps(artifact, allow_nan=False)
        return artifact


def _values(item: Mapping[str, Any]) -> tuple[list[float], list[bool]]:
    raw = item.get("delta")
    raw = raw if isinstance(raw, list) else [raw]
    valid = [item is not None for item in raw]
    return [float(item or 0) for item in raw], valid


def _compare(
    entries: Mapping[str, Mapping[str, Any]], modalities: list[str], common: set[int] | None = None
) -> dict[str, Any]:
    data = {}
    for name in modalities:
        item = entries.get(name, {})
        if item.get("status") == "evaluated":
            values, valid = _values(item)
            data[name] = (values, valid)
    if common is None:
        evaluated = list(data.values())
        common = (
            set.intersection(*({index for index, valid in enumerate(item[1]) if valid} for item in evaluated))
            if evaluated
            else set()
        )
    details = {}
    magnitudes = {}
    for name in modalities:
        item = entries.get(name, {})
        values, valid = data.get(name, ([], []))
        selected = [values[index] for index in sorted(common) if index < len(values) and valid[index]]
        mean = sum(selected) / len(selected) if selected else None
        absolute = sum(abs(value) for value in selected) / len(selected) if selected else None
        if absolute is not None:
            magnitudes[name] = absolute
        details[name] = {
            "status": item.get("status", "already_missing"),
            "mean_signed_delta": mean,
            "mean_absolute_delta": absolute,
            "relative_influence": None,
            "rank": None,
            "valid_comparison_count": sum(valid),
            "common_comparison_count": len(selected),
        }
    ranking = sorted(magnitudes, key=lambda name: (-magnitudes[name], name))
    total = sum(magnitudes.values())
    ties = []
    for name in ranking:
        equal = [other for other in ranking if magnitudes[other] == magnitudes[name]]
        if len(equal) > 1 and equal not in ties:
            ties.append(equal)
    for _index, name in enumerate(ranking):
        details[name]["rank"] = next(
            index2 + 1 for index2, other in enumerate(ranking) if magnitudes[other] == magnitudes[name]
        )
        details[name]["relative_influence"] = magnitudes[name] / total if total else None
    if not ranking:
        status = "unavailable"
    elif total == 0:
        status = "no_measurable_influence"
    else:
        status = "evaluated"
    dominant = (
        ranking[0] if ranking and (len(ranking) == 1 or magnitudes[ranking[0]] > magnitudes[ranking[1]]) else None
    )
    return {
        "status": status,
        "reason": None,
        "ranking": ranking if status != "no_measurable_influence" else [],
        "ties": [] if status == "no_measurable_influence" else ties,
        "dominant_modality": dominant,
        "dominant_modalities": [name for name in ranking if magnitudes.get(name) == magnitudes.get(ranking[0])]
        if ranking
        else [],
        "comparable_modalities": ranking,
        "common_comparison_count": len(common),
        "modalities": details,
    }


def summarize_explanations(explanations: Mapping[str, Any], *, temporal_axis: int | None = None) -> dict[str, Any]:
    """Summarize sample, global, and configured one-dimensional temporal sensitivity."""
    samples = explanations.get("samples")
    modalities = list(explanations.get("modalities") or [])
    if not isinstance(samples, list):
        raise ValueError("explanations['samples'] must be a list.")
    if temporal_axis is not None and temporal_axis != 1:
        raise ValueError("temporal_axis must be a positive batch-relative axis or None.")
    if temporal_axis is not None:
        for sample in samples:
            for item in (sample.get("modalities") or {}).values():
                if (
                    item.get("status") == "evaluated"
                    and item.get("delta") is not None
                    and not isinstance(item.get("delta"), list)
                ):
                    raise ValueError("temporal_axis does not identify a list-like score dimension.")
    sample_summaries = []
    for sample in samples:
        entries = sample.get("modalities") or {}
        summary = _compare(entries, modalities)
        summary["sample_id"] = sample.get("sample_id")
        if temporal_axis is not None:
            temporal = []
            lengths = [len(item.get("delta")) for item in entries.values() if isinstance(item.get("delta"), list)]
            for position in range(max(lengths, default=0)):
                position_entries = {}
                for name in modalities:
                    item = dict(entries.get(name, {}))
                    delta = item.get("delta")
                    item["delta"] = delta[position : position + 1] if isinstance(delta, list) else None
                    position_entries[name] = item
                position_summary = _compare(position_entries, modalities)
                if position_summary["ranking"]:
                    temporal.append({"position": position, **position_summary})
            summary["temporal"] = temporal
        sample_summaries.append(summary)
    candidates = [
        name
        for name in modalities
        if any((sample.get("modalities") or {}).get(name, {}).get("status") == "evaluated" for sample in samples)
    ]
    joint = []
    contributions = {name: [] for name in candidates}
    common_counts = dict.fromkeys(modalities, 0)
    valid_counts = dict.fromkeys(modalities, 0)
    for sample in samples:
        entries = sample.get("modalities") or {}
        data = {}
        for name in modalities:
            item = entries.get(name, {})
            if item.get("status") == "evaluated":
                values, valid = _values(item)
                data[name] = (values, valid)
                valid_counts[name] += sum(valid)
        if candidates and all(name in data for name in candidates):
            common = set.intersection(
                *({index for index, valid in enumerate(data[name][1]) if valid} for name in candidates)
            )
            if common:
                joint.append(sample)
                for name in candidates:
                    selected = [data[name][0][index] for index in common]
                    common_counts[name] += len(selected)
                    contributions[name].append(
                        (sum(selected) / len(selected), sum(abs(value) for value in selected) / len(selected))
                    )
    global_entries = {}
    for name in modalities:
        values = list(contributions.get(name, []))
        global_entries[name] = {
            "status": "evaluated" if values else "already_missing",
            "delta": [item[0] for item in values],
        }
    result = _compare(global_entries, modalities)
    for name in candidates:
        if contributions[name]:
            signed = [item[0] for item in contributions[name]]
            absolute = [item[1] for item in contributions[name]]
            result["modalities"][name]["mean_signed_delta"] = sum(signed) / len(signed)
            result["modalities"][name]["mean_absolute_delta"] = sum(absolute) / len(absolute)
            result["modalities"][name]["valid_comparison_count"] = valid_counts[name]
            result["modalities"][name]["common_comparison_count"] = common_counts[name]
    if candidates:
        magnitudes = {
            name: sum(item[1] for item in contributions[name]) / len(contributions[name])
            for name in candidates
            if contributions[name]
        }
        result["ranking"] = sorted(magnitudes, key=lambda name: (-magnitudes[name], name))
        result["ties"] = [
            [name for name in result["ranking"] if magnitudes[name] == magnitudes[leader]]
            for leader in result["ranking"]
            if sum(magnitudes[name] == magnitudes[leader] for name in result["ranking"]) > 1
        ]
        result["ties"] = [list(group) for group in dict.fromkeys(tuple(group) for group in result["ties"])]
        result["dominant_modality"] = (
            result["ranking"][0]
            if result["ranking"]
            and magnitudes[result["ranking"][0]] > 0
            and (len(result["ranking"]) == 1 or magnitudes[result["ranking"][0]] > magnitudes[result["ranking"][1]])
            else None
        )
        total = sum(magnitudes.values())
        for rank, name in enumerate(result["ranking"], 1):
            result["modalities"][name]["relative_influence"] = magnitudes[name] / total if total else None
            result["modalities"][name]["rank"] = rank
        result["status"] = "evaluated" if total else "no_measurable_influence"
        if not any(magnitudes.values()):
            result["dominant_modalities"] = []
    if candidates and not magnitudes:
        result["status"] = "unavailable"
    result["common_comparison_count"] = max(common_counts.values(), default=0)
    for name in modalities:
        statuses = [
            (sample.get("modalities") or {}).get(name, {}).get("status", "already_missing") for sample in samples
        ]
        result["modalities"][name].update(
            {
                "availability_count": sum(status != "already_missing" for status in statuses),
                "evaluated_sample_count": sum(status == "evaluated" for status in statuses),
                "missing_sample_count": sum(status == "already_missing" for status in statuses),
                "unsupported_sample_count": sum(status == "unsupported" for status in statuses),
            }
        )
    if len(candidates) > 1 and not joint:
        result["status"] = "insufficient_common_support"
        result["ranking"] = []
        result["ties"] = []
        result["dominant_modality"] = None
        result["dominant_modalities"] = []
        for item in result["modalities"].values():
            item["mean_signed_delta"] = None
            item["mean_absolute_delta"] = None
            item["relative_influence"] = None
            item["rank"] = None
    result.update(
        {
            "total_samples": len(samples),
            "comparable_samples": len(joint),
            "sample_coverage": len(joint) / len(samples) if samples else 0.0,
            "samples": sample_summaries,
            "temporal_axis": temporal_axis,
        }
    )
    if temporal_axis is not None:
        result["temporal"] = sample_summaries[0].get("temporal", []) if sample_summaries else []
    json.dumps(result, allow_nan=False)
    return result


@dataclass
class XAIStep:
    """Wrap a prediction step and collect one XAI artifact."""

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
            modalities = list(dict.fromkeys(name for item in self._results for name in item["modalities"]))
            samples = []
            for item in self._results:
                for sample in item["samples"]:
                    sample["modalities"] = {
                        name: sample["modalities"].get(
                            name, _entry("already_missing", "modality was not present in this batch")
                        )
                        for name in modalities
                    }
                    samples.append(sample)
            artifact = {
                "schema_version": 2,
                "method": "leave_one_modality_out",
                "ablation_policy": self.ablation_policy,
                "modalities": modalities,
                "samples": samples,
                "batches": len(self._results),
            }
            artifact["summary"] = summarize_explanations(artifact, temporal_axis=self.temporal_axis)
            result.set_artifact("xai", artifact)
            return result
        finally:
            self.step.on_prediction = previous


@INFERENCE_STEPS.register("xai_step")
def xai_step(step: Mapping[str, Any] | Any, **params: Any) -> XAIStep:
    if isinstance(step, Mapping):
        from chimera_ml.inference.builders import build_inference_step

        step = build_inference_step(dict(step))
    return XAIStep(step=step, **params)
