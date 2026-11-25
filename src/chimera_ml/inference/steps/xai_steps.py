"""Universal XAI wrapper and modality intervention engine."""

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


def _intervene(batch: Batch, modality: str, policy: str) -> Batch:
    if modality not in batch.inputs:
        raise ValueError(f"Modality '{modality}' is not present in the batch.")

    inputs = dict(batch.inputs)
    masks = deepcopy(batch.get_masks() or {})
    meta = deepcopy(batch.meta)
    value = batch.inputs[modality]

    if policy == "zero":
        inputs[modality] = torch.zeros_like(value)
    elif policy == "remove":
        del inputs[modality]
        _drop_modality_masks(masks, modality)
    elif policy == "mask":
        changed = False
        for key in (f"{modality}_mask", f"{modality}_temporal_mask"):
            if key in masks:
                masks[key] = torch.zeros_like(torch.as_tensor(masks[key]).bool())
                changed = True
        if not changed:
            masks[f"{modality}_mask"] = torch.zeros(value.shape[0], dtype=torch.bool, device=value.device)
    else:
        raise ValueError("ablation_policy must be zero, mask, or remove.")

    if policy == "remove" and isinstance(meta, dict) and isinstance(meta.get("masks"), Mapping):
        legacy = dict(meta["masks"])
        _drop_modality_masks(legacy, modality)
        meta["masks"] = legacy

    return Batch(inputs, batch.targets, masks or None, meta)


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

            output = model(intervened)
            changed = _score(output, size, score_fn)
            if changed.shape != original.shape:
                raise ValueError(
                    f"Baseline and ablated scores differ: {tuple(original.shape)} vs {tuple(changed.shape)}."
                )
            changed_valid = _validity(intervened, output, changed, output_mask_key)
            common_valid = baseline_valid & changed_valid
            rejected = torch.zeros(size, dtype=torch.bool, device=common_valid.device)
            if validity_fn is not None:
                valid = validity_fn(name, intervened, output)
                if torch.is_tensor(valid):
                    if valid.ndim == 0 or valid.shape[0] != size:
                        raise ValueError("validity_fn result must be batch-aligned.")
                    valid = valid.to(common_valid.device).bool()
                    while valid.ndim < common_valid.ndim:
                        valid = valid.unsqueeze(-1)
                    valid = valid.expand_as(common_valid)
                    rejected = ~valid.flatten(1).any(dim=1)
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


def evaluate_modality_quality(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    *,
    metric: Callable[[Any, Any], float],
    score_fn: Callable[[ModelOutput], Tensor] | None = None,
    modalities: Iterable[str] | None = None,
    ablation_policy: str = "zero",
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare a held-out metric for baseline and modality interventions."""
    _require_eval(model)
    batches = list(batches)
    names = tuple(modalities or {name for batch in batches for name in batch.inputs})

    def collect(name: str | None) -> tuple[Tensor, Tensor]:
        predictions, targets = [], []
        for batch in batches:
            size = _batch_size(batch)
            output = model(batch if name is None else _intervene(batch, name, ablation_policy))
            predictions.append(_score(output, size, score_fn).cpu())
            targets.append(batch.targets.cpu())
        return torch.cat(predictions), torch.cat(targets)

    with torch.inference_mode():
        baseline_preds, targets = collect(None)
        baseline = float(metric(baseline_preds, targets))
        result = {"baseline": baseline, "higher_is_better": higher_is_better, "modalities": {}}
        for name in names:
            predictions, _ = collect(name)
            quality = float(metric(predictions, targets))
            result["modalities"][name] = {
                "status": "evaluated",
                "quality": quality,
                "signed_quality_change": quality - baseline if higher_is_better else baseline - quality,
            }
        return result


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
        return {
            "schema_version": 1,
            "method": "leave_one_modality_out",
            "ablation_policy": self.ablation_policy,
            "modalities": modalities,
            "samples": samples,
            "batches": len(self._results),
        }


@INFERENCE_STEPS.register("xai_step")
def xai_step(step: Mapping[str, Any] | Any, **params: Any) -> XAIStep:
    """Build XAIStep around another registered inference step."""
    if isinstance(step, Mapping):
        from chimera_ml.inference.builders import build_inference_step

        step = build_inference_step(dict(step))
    return XAIStep(step=step, **params)
