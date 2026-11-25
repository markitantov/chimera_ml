"""Prediction sensitivity after modality masking."""

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.types import ModelOutput


@dataclass(frozen=True)
class ScoreSpec:
    task: str
    kind: str | None = None
    class_index: int | None = None
    output_dim: int = 0
    select_predicted_class: bool = False
    temporal_reduction: str = "mean"

    @classmethod
    def from_value(cls, value: "ScoreSpec | Mapping[str, Any]") -> "ScoreSpec":
        return value if isinstance(value, cls) else cls(**value)


def _ablate(batch: Batch, modality: str) -> Batch:
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    masks = deepcopy(batch.get_masks() or {})
    value = inputs[modality]
    temporal = f"{modality}_temporal_mask"
    presence = f"{modality}_mask"
    if temporal in masks:
        masks[temporal] = torch.zeros_like(torch.as_tensor(masks[temporal]).bool())
    if presence in masks:
        masks[presence] = torch.zeros_like(torch.as_tensor(masks[presence]).bool())
    elif temporal not in masks:
        masks[presence] = torch.zeros(value.shape[0], dtype=torch.bool)
    return Batch(inputs, batch.targets, masks, deepcopy(batch.meta))


def _score(preds: torch.Tensor, spec: ScoreSpec, target_class: torch.Tensor | None, batch: Batch) -> torch.Tensor:
    if spec.task == "classification":
        index = (
            target_class
            if spec.select_predicted_class
            else torch.full((preds.shape[0],), spec.class_index, device=preds.device, dtype=torch.long)
        )
        score = (
            torch.softmax(preds, -1)
            .gather(-1, index.view(-1, *([1] * (preds.ndim - 1))).expand(*preds.shape[:-1], 1))
            .squeeze(-1)
        )
    else:
        score = preds if preds.ndim == 1 else preds[..., spec.output_dim]

    if score.ndim == 2:
        mask = batch.get_masks("sequence_mask")
        if spec.temporal_reduction == "first":
            return score[:, 0]
        if mask is None:
            return score.mean(1) if spec.temporal_reduction == "mean" else score.sum(1)
        mask = torch.as_tensor(mask, device=score.device).bool()
        score = score * mask
        return score.sum(1) / mask.sum(1).clamp_min(1) if spec.temporal_reduction == "mean" else score.sum(1)
    return score


def _json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    return [_json(item) for item in value] if isinstance(value, (list, tuple)) else value


def _diagnostics(output: ModelOutput) -> dict[str, Any]:
    prefixes = (
        "pair_",
        "pair_weights",
        "group_weights",
        "frame_weights",
        "frame_mask",
        "context_mask",
        "context_latent_attention",
        "frame_to_context_attention",
        "anchor",
        "rho_",
        "delta_",
    )
    return {key: _json(value) for key, value in sorted((output.aux or {}).items()) if key.startswith(prefixes)}


def explain_batch(
    model: torch.nn.Module,
    batch: Batch,
    *,
    baseline_output: ModelOutput | None = None,
    score_spec: ScoreSpec | Mapping[str, Any],
    modalities: Iterable[str] | None = None,
    ablation_policy: str = "mask",
) -> dict[str, Any]:
    """Return leave-one-modality-out score changes for one batch."""
    if ablation_policy != "mask":
        raise ValueError("Only ablation_policy='mask' is supported.")
    spec = ScoreSpec.from_value(score_spec)
    names = tuple(sorted(modalities or batch.inputs))
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = baseline_output if baseline_output is not None else model(batch)
            target = baseline.preds.argmax(-1) if spec.select_predicted_class else None
            if target is not None and target.ndim == 2:
                target = target[:, 0]
            original = _score(baseline.preds, spec, target, batch)
            results = {}
            for name in names:
                try:
                    score = _score(model(_ablate(batch, name)).preds, spec, target, batch)
                    results[name] = [
                        {
                            "status": "evaluated",
                            "ablated_score": float(value),
                            "delta": float(change),
                            "absolute_delta": float(change.abs()),
                        }
                        for value, change in zip(score, original - score, strict=True)
                    ]
                except (KeyError, RuntimeError, ValueError) as error:
                    results[name] = [{"status": "unsupported", "reason": str(error)} for _ in original]
            ids = (batch.meta or {}).get("sample_ids", range(len(original)))
            return {
                "schema_version": 1,
                "method": "leave_one_modality_out",
                "score": {
                    "task": spec.task,
                    "kind": spec.kind or ("class_probability" if spec.task == "classification" else "value"),
                    "class_index": spec.class_index,
                    "output_dim": spec.output_dim,
                    "temporal_reduction": spec.temporal_reduction,
                },
                "ablation": {"policy": ablation_policy, "modalities": list(names)},
                "samples": [
                    {
                        "sample_id": str(sample_id),
                        "baseline_score": float(original[index]),
                        "modalities": {name: results[name][index] for name in names},
                        "fusion_diagnostics": _diagnostics(baseline),
                    }
                    for index, sample_id in enumerate(ids)
                ],
            }
    finally:
        model.train(training)


def evaluate_modality_quality(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    *,
    metric: Callable[[Any, Any], float],
    score_spec: ScoreSpec | Mapping[str, Any],
    modalities: Iterable[str] | None = None,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare a held-out metric before and after modality masking."""
    batches = list(batches)
    spec = ScoreSpec.from_value(score_spec)
    names = tuple(sorted(modalities or {name for batch in batches for name in batch.inputs}))

    def collect(name: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        predictions, targets = [], []
        for batch in batches:
            output = model(batch if name is None else _ablate(batch, name))
            predictions.append(_score(output.preds, spec, None, batch).cpu())
            targets.append(batch.targets.cpu())
        return torch.cat(predictions), torch.cat(targets)

    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            predictions, targets = collect(None)
            baseline = float(metric(predictions, targets))
            result = {"baseline": baseline, "higher_is_better": higher_is_better, "modalities": {}}
            for name in names:
                try:
                    predictions, _ = collect(name)
                    quality = float(metric(predictions, targets))
                    result["modalities"][name] = {
                        "status": "evaluated",
                        "quality": quality,
                        "signed_quality_change": quality - baseline if higher_is_better else baseline - quality,
                    }
                except (KeyError, RuntimeError, ValueError) as error:
                    result["modalities"][name] = {"status": "unsupported", "reason": str(error)}
            return result
    finally:
        model.train(training)


def resolve_inference_device(device: str) -> str:
    """Resolve requested inference device to a concrete runtime value."""
    requested = device.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda.")

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    return requested
