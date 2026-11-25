import json

import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import INFERENCE_STEPS
from chimera_ml.core.types import ModelOutput
from chimera_ml.inference.utils import ScoreSpec, explain_batch


class MaskAwareModel(torch.nn.Module):
    def forward(self, batch: Batch) -> ModelOutput:
        values = []
        for name in ("audio", "video"):
            mask = batch.get_masks(f"{name}_mask")
            values.append(batch.inputs[name].flatten(1).sum(1) * torch.as_tensor(mask).to(batch.inputs[name]))
        return ModelOutput(torch.stack(values, dim=-1).sum(dim=-1, keepdim=True))


def test_explain_batch_uses_signed_leave_one_out_delta_without_mutating_batch():
    model = MaskAwareModel()
    batch = Batch(
        inputs={"audio": torch.tensor([[2.0]]), "video": torch.tensor([[3.0]])},
        targets=None,
        masks={"audio_mask": torch.ones(1, dtype=torch.bool), "video_mask": torch.ones(1, dtype=torch.bool)},
        meta={"sample_ids": ["sample-1"]},
    )
    original = {key: value.clone() for key, value in batch.inputs.items()}
    result = explain_batch(model, batch, score_spec=ScoreSpec("regression"), modalities=("audio", "video"))

    assert result["samples"][0]["modalities"]["audio"]["delta"] == 2.0
    assert result["samples"][0]["modalities"]["video"]["delta"] == 3.0
    assert torch.equal(batch.inputs["audio"], original["audio"])
    json.dumps(result, allow_nan=False)


def test_xai_step_is_registered():
    assert "xai_step" in INFERENCE_STEPS._items
