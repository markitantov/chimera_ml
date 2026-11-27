import torch

from chimera_ml.core.batch import Batch
from chimera_ml.inference.steps.xai_steps import explain_batch
from chimera_ml.models.raav_fusion import RAAVFusionModel
from chimera_ml.models.tfs_fusion_model import TFSFusionModel
from chimera_ml.models.tomar_fusion import TOMARFusionModel


class CountingTFS(TFSFusionModel):
    def __init__(self):
        super().__init__({"a": 2, "b": 2, "c": 2}, 2, 1)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return super().forward(batch)


class CountingTOMAR(TOMARFusionModel):
    def __init__(self):
        super().__init__({"anchor": 2, "extra": 2}, 2, "anchor", 1)
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return super().forward(batch)


class CountingRAAV(RAAVFusionModel):
    def __init__(self):
        super().__init__({"frame": 2, "context": 2}, 2, 1, ("frame",), "context")
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        return super().forward(batch)


def test_tfs_remove_is_rejected_before_forward_and_mask_is_supported():
    batch = Batch({name: torch.ones(2, 1, 2) for name in ("a", "b", "c")}, None)
    model = CountingTFS().eval()
    removed = explain_batch(model, batch, modalities=("a",), ablation_policy="remove")
    assert model.calls == 1
    assert removed["samples"][0]["modalities"]["a"]["status"] == "unsupported"

    masked = explain_batch(model, batch, modalities=("a",), ablation_policy="mask")
    assert model.calls == 3
    assert masked["samples"][0]["modalities"]["a"]["status"] == "evaluated"


def test_tomar_anchor_mask_is_rejected_before_forward():
    batch = Batch({"anchor": torch.ones(2, 1, 2), "extra": torch.ones(2, 1, 2)}, None)
    model = CountingTOMAR().eval()
    result = explain_batch(model, batch, modalities=("anchor",), ablation_policy="mask")
    assert model.calls == 1
    assert result["samples"][0]["modalities"]["anchor"]["status"] == "unsupported"


def test_raav_final_frame_removal_is_rejected_before_forward_and_aux_mask_works():
    batch = Batch({"frame": torch.ones(2, 2, 2), "context": torch.ones(2, 1, 2)}, None)
    model = CountingRAAV().eval()
    result = explain_batch(
        model,
        batch,
        modalities=("frame",),
        ablation_policy="remove",
        output_mask_key=lambda _, output: output.aux["frame_mask"],
    )
    assert model.calls == 1
    assert result["samples"][0]["modalities"]["frame"]["status"] == "unsupported"
