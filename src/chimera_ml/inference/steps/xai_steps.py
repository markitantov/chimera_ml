"""Minimal leave-one-modality-out prediction sensitivity analysis."""

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference.context import InferenceContext
from chimera_ml.models._utils import _modality_validity


def _batch_size(batch: Batch) -> int:
    return next(iter(batch.inputs.values())).shape[0]


def _availability(batch: Batch, modality: str, size: int) -> Tensor:
    if modality not in batch.inputs:
        return torch.zeros(size, dtype=torch.bool)

    value = batch.inputs[modality]
    if value.ndim in (2, 3):
        return _modality_validity(batch, modality)

    if value.ndim == 1:
        return torch.ones(value.shape[0], dtype=torch.bool, device=value.device)

    raise ValueError(f"XAI modality availability requires '{modality}' to have shape [B], [B, D], or [B, T, D].")


def _remove_masks(masks: dict[str, Any], modality: str) -> None:
    for key in (modality, f"{modality}_mask", f"{modality}_temporal_mask"):
        masks.pop(key, None)


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
        mask_keys = [key for key in (f"{modality}_mask", f"{modality}_temporal_mask") if key in masks]

        if mask_keys:
            for key in mask_keys:
                masks[key] = torch.zeros_like(torch.as_tensor(masks[key]).bool())
        else:
            masks[f"{modality}_mask"] = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)

    else:
        raise ValueError("ablation_policy must be zero, mask, or remove.")

    # Returning a canonical dictionary, including an empty one, prevents a
    # legacy metadata mask from reappearing through Batch.get_masks().
    canonical_masks = masks if isinstance(batch.masks, dict) or masks else None
    return Batch(inputs, batch.targets, canonical_masks, deepcopy(batch.meta))


def _score(
    output: ModelOutput,
    size: int,
    score_fn: Callable[[ModelOutput], Tensor] | None,
) -> Tensor:
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


def _ids(batch: Batch, batch_index: int | None) -> list[str]:
    raw = (batch.meta or {}).get("sample_ids")
    if raw is not None:
        return [str(value) for value in raw]

    prefix = "sample" if batch_index is None else f"batch-{batch_index}"
    return [f"{prefix}-{index}" for index in range(_batch_size(batch))]


def _entry(status: str, reason: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "ablated": None,
        "delta": None,
        "absolute_delta": None,
    }


def _ensure_finite(value: Tensor) -> None:
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("Out of range float values are not JSON compliant.")


def _masked(value: Tensor, valid: Tensor) -> Any:
    _ensure_finite(value)
    value = value.detach().cpu()
    valid = valid.detach().cpu()

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
        if baseline_output is None:
            baseline_output = model(batch)

        baseline_score = _score(baseline_output, size, score_fn)
        baseline_valid = _validity(batch, baseline_output, baseline_score, output_mask_key)

        diagnostics = {}
        for key in diagnostic_keys or ():
            if baseline_output.aux and key in baseline_output.aux:
                value = baseline_output.aux[key]
                if value.ndim == 0 or value.shape[0] != size:
                    raise ValueError(f"Diagnostic '{key}' must be batch-aligned with size {size}.")
                _ensure_finite(value)
                diagnostics[key] = value

        result: dict[str, list[dict[str, Any]]] = {}
        for modality in names:
            available = _availability(batch, modality, size)
            entries = [_entry("already_missing", "modality is unavailable for this sample") for _ in range(size)]

            if not available.any():
                result[modality] = entries
                continue

            changed_batch = _intervene(batch, modality, ablation_policy)
            validator = getattr(model, "validate_intervention", None)
            reason = None
            if callable(validator):
                reason = validator(batch, changed_batch, modality, ablation_policy)

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

            intervened_valid = _validity(
                changed_batch,
                intervened_output,
                intervened_score,
                output_mask_key,
            )
            valid = baseline_valid & intervened_valid

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
        for index, sample_id in enumerate(_ids(batch, batch_index)):
            sample = {
                "sample_id": sample_id,
                "baseline": _masked(baseline_score[index], baseline_valid[index]),
                "modalities": {name: result[name][index] for name in names},
                "diagnostics": {},
            }

            for key, value in diagnostics.items():
                sample["diagnostics"][key] = value[index].detach().cpu().tolist()

            samples.append(sample)

    return {
        "ablation_policy": ablation_policy,
        "modalities": names,
        "samples": samples,
    }


def _decode_delta(item: Mapping[str, Any]) -> tuple[Tensor, Tensor] | None:
    raw = item.get("delta")
    if raw is None:
        return None

    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        valid = torch.tensor(
            [[value is not None for value in row] for row in raw],
            dtype=torch.bool,
        )
        values = torch.tensor([[float(value) if value is not None else 0.0 for value in row] for row in raw])
    elif isinstance(raw, list):
        valid = torch.tensor(
            [value is not None for value in raw],
            dtype=torch.bool,
        )
        values = torch.tensor([float(value) if value is not None else 0.0 for value in raw])
    else:
        valid = torch.ones(1, dtype=torch.bool)
        values = torch.tensor([float(raw)])

    _ensure_finite(values)
    return values, valid


def _decoded_samples(
    samples: list[Mapping[str, Any]],
    modalities: list[str],
) -> list[tuple[Mapping[str, Any], dict[str, tuple[Tensor, Tensor]]]]:
    decoded = []

    for sample in samples:
        entries = sample.get("modalities") or {}
        sample_entries = {}

        for name in modalities:
            delta = _decode_delta(entries.get(name, {}))
            if delta is not None:
                sample_entries[name] = delta

        decoded.append((sample, sample_entries))

    return decoded


def _summary(
    decoded: list[tuple[Mapping[str, Any], dict[str, tuple[Tensor, Tensor]]]],
    modalities: list[str],
    candidates: list[str],
    position: int | None = None,
) -> dict[str, Any]:
    signed = {name: [] for name in candidates}
    absolute = {name: [] for name in candidates}

    for _, entries in decoded:
        data = {}
        for name in candidates:
            if name not in entries:
                break

            values, valid = entries[name]
            if position is not None:
                if position >= values.shape[0]:
                    break
                values = values[position]
                valid = valid[position]

            data[name] = values, valid

        if len(data) != len(candidates):
            continue

        if len({values.shape for values, _ in data.values()}) != 1:
            continue

        common = torch.stack([valid for _, valid in data.values()]).all(0)
        if not common.any():
            continue

        for name, (values, _) in data.items():
            selected = values[common]
            signed[name].append(selected.mean())
            absolute[name].append(selected.abs().mean())

    means = {}
    for name in candidates:
        if signed[name]:
            means[name] = (
                torch.stack(signed[name]).mean(),
                torch.stack(absolute[name]).mean(),
            )

    ranking = sorted(means, key=lambda name: (-float(means[name][1]), name))
    total = sum(float(means[name][1]) for name in ranking)
    summary_modalities = {}

    for name in modalities:
        if name in means:
            signed_mean, absolute_mean = means[name]
            relative = float(absolute_mean) / total if total else None
            summary_modalities[name] = {
                "mean_signed_delta": float(signed_mean),
                "mean_absolute_delta": float(absolute_mean),
                "relative_sensitivity": relative,
            }
        else:
            summary_modalities[name] = {
                "mean_signed_delta": None,
                "mean_absolute_delta": None,
                "relative_sensitivity": None,
            }

    return {"ranking": ranking, "modalities": summary_modalities}


def summarize_explanations(
    explanations: Mapping[str, Any],
    *,
    temporal_axis: int | None = None,
) -> dict[str, Any]:
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
            lengths = []
            for name, (values, _) in entries.items():
                if isinstance(raw_entries.get(name, {}).get("delta"), list):
                    lengths.append(values.shape[0])

            item["temporal"] = []
            for position in range(max(lengths, default=0)):
                position_result = _summary([(sample, entries)], modalities, names, position)
                if position_result["ranking"]:
                    item["temporal"].append({"position": position, **position_result})

        result["samples"].append(item)

    if temporal_axis is not None:
        lengths = [values.shape[0] for _, entries in decoded for values, _ in entries.values()]
        result["temporal"] = []

        for position in range(max(lengths, default=0)):
            position_result = _summary(decoded, modalities, candidates, position)
            if position_result["ranking"]:
                result["temporal"].append({"position": position, **position_result})

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

    def run(self, ctx: InferenceContext) -> InferenceContext:
        if not hasattr(self.step, "on_prediction"):
            raise TypeError(
                f"Wrapped step '{type(self.step).__name__}' must expose an 'on_prediction' callback attribute."
            )

        previous = self.step.on_prediction
        if previous is not None and not callable(previous):
            raise TypeError("Wrapped step 'on_prediction' must be callable or None.")

        results = []

        def collect(model: nn.Module, batch: Batch, output: ModelOutput) -> None:
            if previous is not None:
                previous(model, batch, output)

            results.append(
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
                    batch_index=len(results),
                )
            )

        self.step.on_prediction = collect

        try:
            result = self.step.run(ctx)
            modalities = []
            for item in results:
                for name in item["modalities"]:
                    if name not in modalities:
                        modalities.append(name)

            samples = []
            for item in results:
                for sample in item["samples"]:
                    complete_modalities = {}
                    for name in modalities:
                        complete_modalities[name] = sample["modalities"].get(
                            name,
                            _entry(
                                "already_missing",
                                "modality was not present in this batch",
                            ),
                        )
                    sample["modalities"] = complete_modalities
                    samples.append(sample)

            artifact = {
                "ablation_policy": self.ablation_policy,
                "modalities": modalities,
                "samples": samples,
            }
            artifact["summary"] = summarize_explanations(
                artifact,
                temporal_axis=self.temporal_axis,
            )
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
