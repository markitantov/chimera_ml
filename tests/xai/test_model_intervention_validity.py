import torch

from chimera_ml.core.batch import Batch
from chimera_ml.inference.steps.xai_steps import explain_batch
from chimera_ml.models.raav_fusion_model import RAAVFusionModel
from chimera_ml.models.tfs_fusion_model import TFSFusionModel
from chimera_ml.models.tomar_fusion_model import TOMARFusionModel


def test_tfs_remove_and_mask_keep_a_supported_pair() -> None:
    batch = Batch({name: torch.ones(2, 1, 2) for name in ("a", "b", "c")}, None)
    model = TFSFusionModel({"a": 2, "b": 2, "c": 2}, 2, 1).eval()

    removed = explain_batch(model, batch, modalities=("a",), ablation_policy="remove")
    assert removed["samples"][0]["modalities"]["a"]["status"] == "evaluated"

    masked = explain_batch(model, batch, modalities=("a",), ablation_policy="mask")
    assert masked["samples"][0]["modalities"]["a"]["status"] == "evaluated"


def test_tomar_anchor_mask_is_unsupported() -> None:
    batch = Batch({"anchor": torch.ones(2, 1, 2), "extra": torch.ones(2, 1, 2)}, None)
    model = TOMARFusionModel({"anchor": 2, "extra": 2}, 2, "anchor", 1).eval()
    result = explain_batch(model, batch, modalities=("anchor",), ablation_policy="mask")

    assert result["samples"][0]["modalities"]["anchor"]["status"] == "unsupported"


def test_raav_context_intervention_returns_sequence_score() -> None:
    batch = Batch({"frame": torch.ones(2, 2, 2), "context": torch.ones(2, 1, 2)}, None)
    model = RAAVFusionModel(
        {"frame": 2, "context": 2},
        2,
        1,
        ("frame",),
        ("context",),
    ).eval()
    result = explain_batch(model, batch, modalities=("context",))

    assert result["samples"][0]["modalities"]["context"]["status"] == "evaluated"
