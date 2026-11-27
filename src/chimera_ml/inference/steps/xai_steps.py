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
from chimera_ml.core.types import ModelOutput
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
    present = (
        torch.ones((size, width), dtype=torch.bool, device=value.device)
        if mask is None
        else torch.as_tensor(mask, device=value.device).bool()
    )
    if present.ndim == 1:
        present = present[:, None].expand(-1, width)
    sequence = batch.get_masks("sequence_mask")
    if sequence is not None:
        sequence = torch.as_tensor(sequence, device=value.device).bool()
        sequence = sequence[:, None] if sequence.ndim == 1 else sequence
        if sequence.shape == present.shape:
            present = present & sequence
    return present.any(1)


def _remove_masks(masks: dict[str, Any], modality: str) -> None:
    keys = (modality, f"{modality}_mask", f"{modality}_temporal_mask")
    for key in keys:
        masks.pop(key, None)
    nested = masks.get("mask")
    if isinstance(nested, dict):
        nested = dict(nested)
        for key in keys:
            nested.pop(key, None)
        masks["mask"] = nested


def _intervene(batch: Batch, modality: str, policy: str) -> Batch:
    if modality not in batch.inputs:
        raise ValueError(f"Modality '{modality}' is not present in the batch.")
    inputs = dict(batch.inputs)
    masks = deepcopy(batch.get_masks() or {})
    value = inputs[modality]
    if policy == "zero":
        inputs[modality] = torch.zeros_like(value)
    elif policy == "remove":
        del inputs[modality]
        _remove_masks(masks, modality)
    elif policy == "mask":
        keys = [key for key in (f"{modality}_mask", f"{modality}_temporal_mask") if key in masks]
        if keys:
            for key in keys:
                masks[key] = torch.zeros_like(torch.as_tensor(masks[key]).bool())
        else:
            masks[f"{modality}_mask"] = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)
    else:
        raise ValueError("ablation_policy must be zero, mask, or remove.")
    meta = deepcopy(batch.meta)
    if policy == "remove" and isinstance(meta, dict) and isinstance(meta.get("masks"), Mapping):
        _remove_masks(meta["masks"], modality)
    return Batch(inputs, batch.targets, masks or None, meta)


def _score(output: ModelOutput, size: int, score_fn: Callable[[ModelOutput], Tensor] | None) -> Tensor:
    score = output.preds if score_fn is None else score_fn(output)
    if not torch.is_tensor(score):
        raise TypeError("score_fn must return a torch.Tensor.")
    if score.ndim == 0 or score.shape[0] != size:
        raise ValueError("Scores must be batch-aligned with the input batch.")
    return score


def _validity(
    batch: Batch, output: ModelOutput, score: Tensor, key: str | Callable[[Batch, ModelOutput], Tensor] | None
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


def _ids(batch: Batch, batch_index: int | None) -> list[str]:
    raw = (batch.meta or {}).get("sample_ids")
    if raw is not None:
        return [str(value) for value in raw]
    prefix = "sample" if batch_index is None else f"batch-{batch_index}"
    return [f"{prefix}-{index}" for index in range(_batch_size(batch))]


def _entry(status: str, reason: str | None = None) -> dict[str, Any]:
    return {"status": status, "reason": reason, "ablated": None, "delta": None, "absolute_delta": None}


def _masked(value: Tensor, valid: Tensor) -> Any:
    value, valid = value.detach().cpu(), valid.detach().cpu()
    if value.ndim == 0:
        return value.item() if bool(valid) else None
    if value.ndim == 1:
        return [item if keep else None for item, keep in zip(value.tolist(), valid.tolist(), strict=True)]
    if value.ndim == 2:
        return [
            [item if keep else None for item, keep in zip(row, row_valid, strict=True)]
            for row, row_valid in zip(value.tolist(), valid.tolist(), strict=True)
        ]
    raise ValueError("Supported scores have at most two non-batch dimensions.")


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
        diagnostics = {
            key: baseline_output.aux[key]
            for key in diagnostic_keys or ()
            if baseline_output.aux and key in baseline_output.aux
        }
        for key, value in diagnostics.items():
            if value.ndim == 0 or value.shape[0] != size:
                raise ValueError(f"Diagnostic '{key}' must be batch-aligned with size {size}.")
        result: dict[str, list[dict[str, Any]]] = {}
        for modality in names:
            available = _availability(batch, modality, size)
            entries = [_entry("already_missing", "modality is unavailable for this sample") for _ in range(size)]
            if not available.any():
                result[modality] = entries
                continue
            changed_batch = _intervene(batch, modality, ablation_policy)
            validator = getattr(model, "validate_intervention", None)
            reason = validator(batch, changed_batch, modality, ablation_policy) if callable(validator) else None
            if reason is not None and not isinstance(reason, str):
                raise TypeError("validate_intervention must return str or None.")
            if reason is not None:
                for index in available.nonzero().flatten().tolist():
                    entries[index] = _entry("unsupported", reason)
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
                    text = (
                        "validity_fn rejected the ablated sample"
                        if validity_fn is not None
                        else "no common valid output positions remain"
                    )
                    entries[index] = _entry("unsupported", text)
                else:
                    entries[index] = {
                        "status": "evaluated",
                        "reason": None,
                        "ablated": _masked(intervened_score[index], sample_valid),
                        "delta": _masked(delta[index], sample_valid),
                        "absolute_delta": _masked(delta[index].abs(), sample_valid),
                    }
            result[modality] = entries
        samples = [
            {
                "sample_id": sample_id,
                "baseline": _masked(baseline_score[index], baseline_valid[index]),
                "modalities": {name: result[name][index] for name in names},
                "diagnostics": {key: value[index].detach().cpu().tolist() for key, value in diagnostics.items()},
            }
            for index, sample_id in enumerate(_ids(batch, batch_index))
        ]
    artifact = {
        "schema_version": 2,
        "method": "leave_one_modality_out",
        "ablation_policy": ablation_policy,
        "modalities": names,
        "samples": samples,
    }
    json.dumps(artifact, allow_nan=False)
    return artifact


def _decode_delta(item: Mapping[str, Any]) -> tuple[Tensor, Tensor] | None:
    raw = item.get("delta")
    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        valid = torch.tensor([[value is not None for value in row] for row in raw], dtype=torch.bool)
        values = torch.tensor([[float(value) if value is not None else 0.0 for value in row] for row in raw])
    elif isinstance(raw, list):
        valid = torch.tensor([value is not None for value in raw], dtype=torch.bool)
        values = torch.tensor([float(value) if value is not None else 0.0 for value in raw])
    else:
        valid, values = torch.ones(1, dtype=torch.bool), torch.tensor([float(raw)])
    return values, valid


def _decoded_samples(
    samples: list[Mapping[str, Any]], modalities: list[str]
) -> list[tuple[Mapping[str, Any], dict[str, tuple[Tensor, Tensor]]]]:
    decoded = []
    for sample in samples:
        entries = sample.get("modalities") or {}
        decoded.append(
            (sample, {name: delta for name in modalities if (delta := _decode_delta(entries.get(name, {})))})
        )
    return decoded


def _summary(
    decoded: list[tuple[Mapping[str, Any], dict[str, tuple[Tensor, Tensor]]]],
    modalities: list[str],
    candidates: list[str],
    position: int | None = None,
) -> dict[str, Any]:
    signed, absolute = {name: [] for name in candidates}, {name: [] for name in candidates}
    for _, entries in decoded:
        data = {}
        for name in candidates:
            if name not in entries:
                break
            values, valid = entries[name]
            if position is not None:
                if position >= values.shape[0]:
                    break
                values, valid = values[position], valid[position]
            data[name] = values, valid
        if len(data) != len(candidates) or len({values.shape for values, _ in data.values()}) != 1:
            continue
        common = torch.stack([valid for _, valid in data.values()]).all(0)
        if not common.any():
            continue
        for name, (values, _) in data.items():
            selected = values[common]
            signed[name].append(selected.mean())
            absolute[name].append(selected.abs().mean())
    means = {
        name: (torch.stack(signed[name]).mean(), torch.stack(absolute[name]).mean())
        for name in candidates
        if signed[name]
    }
    ranking = sorted(means, key=lambda name: (-float(means[name][1]), name))
    total = sum(float(means[name][1]) for name in ranking)
    return {
        "ranking": ranking,
        "modalities": {
            name: {
                "mean_signed_delta": float(means[name][0]) if name in means else None,
                "mean_absolute_delta": float(means[name][1]) if name in means else None,
                "relative_sensitivity": float(means[name][1]) / total if name in means and total else None,
                "rank": ranking.index(name) + 1 if name in ranking else None,
            }
            for name in modalities
        },
    }


def summarize_explanations(explanations: Mapping[str, Any], *, temporal_axis: int | None = None) -> dict[str, Any]:
    """Summarize signed, absolute, and relative prediction sensitivity."""
    samples = explanations.get("samples")
    modalities = list(explanations.get("modalities") or [])
    if not isinstance(samples, list):
        raise ValueError("explanations['samples'] must be a list.")
    if temporal_axis not in (None, 1):
        raise ValueError("temporal_axis must be a positive batch-relative axis or None.")
    decoded = _decoded_samples(samples, modalities)
    if temporal_axis is not None:
        for sample in samples:
            for entry in (sample.get("modalities") or {}).values():
                if entry.get("status") == "evaluated" and not isinstance(entry.get("delta"), list):
                    raise ValueError("temporal_axis does not identify a list-like score dimension.")
    candidates = [name for name in modalities if any(name in entries for _, entries in decoded)]
    result = _summary(decoded, modalities, candidates)
    result["samples"] = []
    for sample, entries in decoded:
        names = [name for name in modalities if name in entries]
        item = _summary([(sample, entries)], modalities, names)
        item["sample_id"] = sample.get("sample_id")
        if temporal_axis is not None:
            raw_entries = sample.get("modalities") or {}
            lengths = [
                values.shape[0]
                for name, (values, _) in entries.items()
                if isinstance(raw_entries.get(name, {}).get("delta"), list)
            ]
            item["temporal"] = [
                {"position": position, **position_result}
                for position in range(max(lengths, default=0))
                if (position_result := _summary([(sample, entries)], modalities, names, position))["ranking"]
            ]
        result["samples"].append(item)
    if temporal_axis is not None:
        lengths = [values.shape[0] for _, entries in decoded for values, _ in entries.values()]
        result["temporal"] = [
            {"position": position, **position_result}
            for position in range(max(lengths, default=0))
            if (position_result := _summary(decoded, modalities, candidates, position))["ranking"]
        ]
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
                "schema_version": 3,
                "method": "leave_one_modality_out",
                "ablation_policy": self.ablation_policy,
                "modalities": modalities,
                "samples": samples,
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
