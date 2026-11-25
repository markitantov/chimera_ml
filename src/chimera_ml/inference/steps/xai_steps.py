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


def _intervene(batch: Batch, modality: str, policy: str) -> Batch:
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    masks = deepcopy(batch.get_masks() or {})
    if modality not in inputs:
        raise ValueError(f"Modality '{modality}' is not present in the batch.")

    if policy == "zero":
        inputs[modality] = torch.zeros_like(inputs[modality])
    elif policy == "remove":
        del inputs[modality]
    elif policy == "mask":
        temporal = f"{modality}_temporal_mask"
        presence = f"{modality}_mask"
        if temporal in masks:
            masks[temporal] = torch.zeros_like(torch.as_tensor(masks[temporal]).bool())
        if presence in masks:
            masks[presence] = torch.zeros_like(torch.as_tensor(masks[presence]).bool())
        elif temporal not in masks:
            masks[presence] = torch.zeros(inputs[modality].shape[0], dtype=torch.bool)
    else:
        raise ValueError("ablation_policy must be 'zero', 'mask', or 'remove'.")
    return Batch(inputs, batch.targets, masks, deepcopy(batch.meta))


def _json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


def _sample(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index] if value.ndim else value
    if isinstance(value, Mapping):
        return {key: _sample(item, index) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _score(
    output: ModelOutput, batch: Batch, score_fn: Callable[[torch.Tensor], Any] | None, mask_key: str | None
) -> Any:
    score = score_fn(output.preds) if score_fn is not None else output.preds
    if not torch.is_tensor(score):
        score = torch.as_tensor(score)
    if mask_key is not None:
        mask = torch.as_tensor(batch.get_masks(mask_key), device=score.device).bool()
        while mask.ndim < score.ndim:
            mask = mask.unsqueeze(-1)
        score = score.masked_fill(~mask, 0)
    return score


def explain_batch(
    model: torch.nn.Module,
    batch: Batch,
    *,
    baseline_output: ModelOutput | None = None,
    score_fn: Callable[[torch.Tensor], Any] | None = None,
    modalities: Iterable[str] | None = None,
    ablation_policy: str = "zero",
    diagnostic_keys: Iterable[str] | None = None,
    output_mask_key: str | None = None,
    validity_fn: Callable[[str, Batch, ModelOutput], bool] | None = None,
) -> dict[str, Any]:
    """Compare a baseline ModelOutput with one intervention per modality."""
    if ablation_policy not in {"zero", "mask", "remove"}:
        raise ValueError("ablation_policy must be zero, mask, or remove.")
    names = tuple(modalities or batch.inputs)
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            baseline = baseline_output if baseline_output is not None else model(batch)
            original = _score(baseline, batch, score_fn, output_mask_key)
            diagnostics = {
                key: baseline.aux[key] for key in sorted(diagnostic_keys or ()) if baseline.aux and key in baseline.aux
            }
            results = {}
            for name in names:
                try:
                    intervened = _intervene(batch, name, ablation_policy)
                except ValueError as error:
                    results[name] = [{"status": "unsupported", "reason": str(error)} for _ in original]
                    continue
                output = model(intervened)
                if validity_fn is not None and not validity_fn(name, intervened, output):
                    results[name] = [
                        {"status": "unsupported", "reason": "validity_fn returned false"} for _ in original
                    ]
                    continue
                changed = _score(output, intervened, score_fn, output_mask_key)
                if changed.shape != original.shape:
                    raise ValueError(
                        f"Baseline and ablated scores differ: {tuple(original.shape)} vs {tuple(changed.shape)}."
                    )
                delta = original - changed
                results[name] = [
                    {
                        "status": "evaluated",
                        "ablated": _json(_sample(changed, index)),
                        "delta": _json(_sample(delta, index)),
                        "absolute_delta": _json(_sample(delta.abs(), index)),
                    }
                    for index in range(original.shape[0])
                ]

            sample_ids = (batch.meta or {}).get("sample_ids", range(original.shape[0]))
            samples = [
                {
                    "sample_id": str(sample_id),
                    "baseline": _json(_sample(original, index)),
                    "modalities": {name: results[name][index] for name in names},
                    "diagnostics": {key: _sample(value, index) for key, value in diagnostics.items()},
                }
                for index, sample_id in enumerate(sample_ids)
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
    finally:
        model.train(training)


def evaluate_modality_quality(
    model: torch.nn.Module,
    batches: Iterable[Batch],
    *,
    metric: Callable[[Any, Any], float],
    score_fn: Callable[[torch.Tensor], Any] | None = None,
    modalities: Iterable[str] | None = None,
    ablation_policy: str = "zero",
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare a held-out metric for baseline and modality interventions."""
    batches = list(batches)
    names = tuple(modalities or {name for batch in batches for name in batch.inputs})

    def collect(name: str | None) -> tuple[torch.Tensor, torch.Tensor]:
        predictions, targets = [], []
        for batch in batches:
            output = model(batch if name is None else _intervene(batch, name, ablation_policy))
            predictions.append(_score(output, batch, score_fn, None).cpu())
            targets.append(batch.targets.cpu())
        return torch.cat(predictions), torch.cat(targets)

    training = model.training
    model.eval()
    try:
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
    finally:
        model.train(training)


@dataclass
class XAIStep:
    """Run a callback-capable prediction step and collect explanations."""

    step: Any
    ablation_policy: str = "zero"
    modalities: Iterable[str] | None = None
    score_fn: Callable[[Tensor], Any] | None = None
    diagnostic_keys: Iterable[str] | None = None
    output_mask_key: str | None = None
    validity_fn: Callable[[str, Batch, ModelOutput], bool] | None = None
    _results: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def run(self, ctx: InferenceContext) -> InferenceContext:
        if not hasattr(self.step, "on_prediction"):
            raise TypeError(
                f"Wrapped step '{type(self.step).__name__}' must expose an 'on_prediction' callback attribute."
            )

        self._results = []
        previous = self.step.on_prediction

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
                )
            )

        self.step.on_prediction = collect
        try:
            result = self.step.run(ctx)
        finally:
            self.step.on_prediction = previous

        ctx.set_artifact("xai", self._combine())
        return result

    def _combine(self) -> dict[str, Any]:
        if not self._results:
            return {
                "schema_version": 1,
                "method": "leave_one_modality_out",
                "ablation_policy": self.ablation_policy,
                "modalities": list(self.modalities or ()),
                "samples": [],
            }
        result = dict(self._results[0])
        result["samples"] = [sample for item in self._results for sample in item["samples"]]
        result["batches"] = len(self._results)
        return result


@INFERENCE_STEPS.register("xai_step")
def xai_step(step: Mapping[str, Any] | Any, **params: Any) -> XAIStep:
    """Build XAIStep around another registered inference step."""
    if isinstance(step, Mapping):
        from chimera_ml.inference.builders import build_inference_step

        step = build_inference_step(dict(step))
    return XAIStep(step=step, **params)
