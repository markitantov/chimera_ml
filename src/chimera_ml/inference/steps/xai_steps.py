from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference.context import InferenceContext


class ScoreFunction(Protocol):
    def __call__(self, output: ModelOutput) -> torch.Tensor: ...


def modality_available(batch: Batch, name: str) -> torch.Tensor:
    values = batch.inputs.get(name)
    if values is None:
        return torch.zeros(0, dtype=torch.bool)

    batch_size = values.shape[0]
    time_steps = 1 if values.ndim == 2 else values.shape[1]
    temporal = batch.get_masks(f"{name}_temporal_mask")

    if temporal is not None:
        mask = torch.as_tensor(temporal, device=values.device).bool()
        if mask.shape != (batch_size, time_steps):
            raise ValueError(f"Temporal mask for '{name}' must have shape {(batch_size, time_steps)}.")
    else:
        presence = batch.get_masks(f"{name}_mask")
        if presence is None:
            mask = torch.ones((batch_size, time_steps), dtype=torch.bool, device=values.device)
        else:
            mask = torch.as_tensor(presence, device=values.device).bool()
            if mask.ndim == 1 and mask.shape[0] == batch_size:
                mask = mask[:, None].expand(-1, time_steps)
            elif mask.shape != (batch_size, time_steps):
                raise ValueError(f"Mask for '{name}' must have shape (B,) or (B, T).")

    sequence = batch.get_masks("sequence_mask")
    if sequence is not None:
        sequence = torch.as_tensor(sequence, device=values.device).bool()
        if sequence.shape == mask.shape:
            mask = mask & sequence

    return mask.any(dim=1)


def intervene_batch(batch: Batch, modality: str, policy: str) -> Batch:
    if policy not in {"remove", "mask"}:
        raise ValueError("ablation_policy must be one of: remove, mask.")

    changed = deepcopy(batch)
    if modality not in changed.inputs:
        return changed

    mask_keys = (modality, f"{modality}_mask", f"{modality}_temporal_mask")
    mask_maps: list[dict[str, Any]] = []
    if isinstance(changed.masks, dict):
        mask_maps.append(changed.masks)
        nested_masks = changed.masks.get("mask")
        if isinstance(nested_masks, dict):
            mask_maps.append(nested_masks)

    legacy_masks = (changed.meta or {}).get("masks")
    if isinstance(legacy_masks, dict):
        if legacy_masks is not changed.masks:
            mask_maps.append(legacy_masks)

        nested_masks = legacy_masks.get("mask")
        if isinstance(nested_masks, dict):
            mask_maps.append(nested_masks)

    if policy == "remove":
        changed.inputs.pop(modality)

        for masks in mask_maps:
            for key in mask_keys:
                masks.pop(key, None)

        return changed

    values = changed.inputs[modality]
    if values.ndim == 2:
        mask_shape = (values.shape[0],)
    elif values.ndim == 3:
        mask_shape = (values.shape[0], values.shape[1])
    else:
        raise ValueError(f"Expected [B, D] or [B, T, D], got {tuple(values.shape)}.")

    changed_mask = False
    for masks in mask_maps:
        for key in mask_keys:
            if key in masks:
                masks[key] = torch.zeros_like(masks[key], dtype=torch.bool)
                changed_mask = True

    if not changed_mask:
        if not mask_maps:
            changed.masks = {}
            mask_maps.append(changed.masks)

        mask_maps[0][f"{modality}_mask"] = torch.zeros(mask_shape, dtype=torch.bool, device=values.device)

    return changed


def select_score(output: ModelOutput, score_fn: ScoreFunction | None) -> torch.Tensor:
    score = output.preds if score_fn is None else score_fn(output)
    if not torch.is_tensor(score):
        raise TypeError("score_fn must return a torch.Tensor.")

    return score


def score_mask(
    batch: Batch,
    model: nn.Module,
    output: ModelOutput,
    score: torch.Tensor,
    output_mask: str | Callable[[nn.Module, ModelOutput], torch.Tensor] | None,
) -> torch.Tensor:
    if output_mask is None:
        return torch.ones_like(score, dtype=torch.bool)

    raw = batch.get_masks(output_mask) if isinstance(output_mask, str) else output_mask(model, output)
    mask = torch.as_tensor(raw, device=score.device).bool()

    if mask.shape == score.shape:
        return mask

    if score.ndim > mask.ndim and mask.shape == score.shape[:-1]:
        return mask.unsqueeze(-1).expand_as(score)

    raise ValueError("output_mask must match the selected score shape.")


def json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("XAI output contains a non-finite value.")

        return value.tolist()

    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]

    return value


def masked_value(value: torch.Tensor, valid: torch.Tensor) -> Any:
    if valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1).expand_as(value)

    if value.ndim == 0:
        return json_value(value) if bool(valid) else None

    return [masked_value(item, item_valid) for item, item_valid in zip(value, valid, strict=True)]


@torch.inference_mode()
def explain_batch(
    model: nn.Module,
    batch: Batch,
    *,
    modalities: tuple[str, ...] | None = None,
    score_fn: ScoreFunction | None = None,
    output_mask: str | Callable[[nn.Module, ModelOutput], torch.Tensor] | None = None,
    ablation_policy: str = "remove",
    baseline_output: ModelOutput | None = None,
    batch_index: int = 0,
) -> dict[str, Any]:
    if model.training:
        raise ValueError("XAI requires the model to be in evaluation mode.")

    if any(module.training for module in model.modules()):
        raise ValueError("XAI requires all nested modules to be in evaluation mode.")

    if not batch.inputs:
        raise ValueError("XAI requires at least one input modality.")

    selected = tuple(batch.inputs) if modalities is None else tuple(modalities)
    baseline = model(batch) if baseline_output is None else baseline_output
    baseline_score = select_score(baseline, score_fn)
    batch_size = next(iter(batch.inputs.values())).shape[0]

    if baseline_score.ndim == 0 or baseline_score.shape[0] != batch_size:
        raise ValueError("Selected score must be batch-aligned.")

    baseline_valid = score_mask(batch, model, baseline, baseline_score, output_mask)
    sample_ids = (batch.meta or {}).get("sample_ids")
    if isinstance(sample_ids, (list, tuple)) and len(sample_ids) == batch_size:
        sample_ids = [str(sample_id) for sample_id in sample_ids]
    else:
        sample_ids = [f"batch-{batch_index}-{index}" for index in range(batch_size)]
    explanations = [
        {
            "sample_id": sample_ids[index],
            "baseline": masked_value(baseline_score[index], baseline_valid[index]),
            "modalities": {},
            "aux": {},
        }
        for index in range(batch_size)
    ]

    if baseline.aux is not None:
        for key, value in baseline.aux.items():
            if value.shape[0] != batch_size:
                continue

            for index in range(batch_size):
                explanations[index]["aux"][key] = json_value(value[index])

    for name in selected:
        available = modality_available(batch, name)
        intervention = intervene_batch(batch, name, ablation_policy)
        validator = getattr(model, "validate_intervention", None)
        reason = validator(batch, intervention, name, ablation_policy) if callable(validator) else None

        if reason is not None:
            for explanation in explanations:
                explanation["modalities"][name] = {
                    "status": "unsupported",
                    "reason": reason,
                    "delta": None,
                }

            continue

        if not bool(available.any()):
            for explanation in explanations:
                explanation["modalities"][name] = {
                    "status": "already_missing",
                    "reason": None,
                    "delta": None,
                }

            continue

        ablated = model(intervention)
        ablated_score = select_score(ablated, score_fn)
        if ablated_score.shape != baseline_score.shape:
            raise ValueError("Ablated score must have the same shape as the baseline score.")

        ablated_valid = score_mask(intervention, model, ablated, ablated_score, output_mask)
        valid = baseline_valid & ablated_valid
        delta = baseline_score - ablated_score

        for index in range(batch_size):
            if not bool(available[index]):
                status = "already_missing"
                value = None
            elif not bool(valid[index].any()):
                status = "unsupported"
                value = None
            else:
                status = "evaluated"
                value = masked_value(delta[index], valid[index])

            explanations[index]["modalities"][name] = {
                "status": status,
                "reason": None if status != "unsupported" else "no common valid output positions",
                "delta": value,
            }

    return {
        "ablation_policy": ablation_policy,
        "modalities": list(selected),
        "samples": explanations,
    }


@dataclass
class XAIStep:
    """Collect leave-one-modality-out prediction sensitivity."""

    step: Any
    modalities: tuple[str, ...] | None = None
    score_fn: ScoreFunction | None = None
    output_mask: str | Callable[[nn.Module, ModelOutput], torch.Tensor] | None = None
    ablation_policy: str = "remove"

    def run(self, ctx: InferenceContext) -> InferenceContext:
        callback = getattr(self.step, "on_prediction", None)
        if callback is not None and not callable(callback):
            raise TypeError("on_prediction must be callable or None.")

        samples: list[dict[str, Any]] = []
        used_modalities: list[str] = []
        seen_modalities: set[str] = set()
        for name in self.modalities or ():
            if name not in seen_modalities:
                used_modalities.append(name)
                seen_modalities.add(name)

        batch_index = 0

        def collect(model: nn.Module, batch: Batch, output: ModelOutput) -> None:
            nonlocal batch_index
            result = explain_batch(
                model,
                batch,
                modalities=self.modalities,
                score_fn=self.score_fn,
                output_mask=self.output_mask,
                ablation_policy=self.ablation_policy,
                baseline_output=output,
                batch_index=batch_index,
            )
            samples.extend(result["samples"])
            for name in result["modalities"]:
                if name not in seen_modalities:
                    used_modalities.append(name)
                    seen_modalities.add(name)

            batch_index += 1

            if callback is not None:
                callback(model, batch, output)

        self.step.on_prediction = collect
        try:
            result = self.step.run(ctx)
        finally:
            self.step.on_prediction = callback

        result.set_artifact(
            "xai",
            {
                "ablation_policy": self.ablation_policy,
                "modalities": used_modalities,
                "samples": samples,
            },
        )
        return result


@INFERENCE_STEPS.register("xai_step")
def xai_step(**params: Any) -> XAIStep:
    step = params.get("step")
    if isinstance(step, dict):
        from chimera_ml.inference.builders import build_inference_step

        params["step"] = build_inference_step(step)

    return XAIStep(**params)
