import torch

from chimera_ml.core.batch import Batch
from chimera_ml.inference.steps.xai_steps import explain_batch
from chimera_ml.models.raav_fusion_model import RAAVFusionModel
from chimera_ml.models.tfs_fusion_model import TFSFusionModel
from chimera_ml.models.tomar_fusion_model import TOMARFusionModel


def test_tfs_rejects_intervention_without_a_directed_pair() -> None:
    batch = Batch(
        {"left": torch.ones(2, 1, 2), "right": torch.ones(2, 1, 2)},
        None,
    )
    model = TFSFusionModel({"left": 2, "right": 2}, 2, 1).eval()

    result = explain_batch(model, batch, modalities=("left",), ablation_policy="remove")

    assert result["samples"][0]["modalities"]["left"]["status"] == "unsupported"
    assert result["samples"][0]["modalities"]["left"]["reason"] == "no configured directed pair remains available"


def test_tfs_keeps_another_pair_after_intervention() -> None:
    batch = Batch(
        {name: torch.ones(2, 1, 2) for name in ("a", "b", "c")},
        None,
    )
    model = TFSFusionModel({"a": 2, "b": 2, "c": 2}, 2, 1).eval()

    result = explain_batch(model, batch, modalities=("a",), ablation_policy="remove")

    assert result["samples"][0]["modalities"]["a"]["status"] == "evaluated"


def test_raav_rejects_removing_the_last_frame_modality() -> None:
    batch = Batch(
        {"frame": torch.ones(2, 2, 2), "context": torch.ones(2, 1, 2)},
        None,
    )
    model = RAAVFusionModel(
        {"frame": 2, "context": 2},
        2,
        1,
        ("frame",),
        ("context",),
    ).eval()

    result = explain_batch(model, batch, modalities=("frame",), ablation_policy="mask")

    assert result["samples"][0]["modalities"]["frame"]["status"] == "unsupported"
    assert result["samples"][0]["modalities"]["frame"]["reason"] == "no usable frame modality remains"


def test_raav_allows_context_intervention() -> None:
    batch = Batch(
        {"frame": torch.ones(2, 2, 2), "context": torch.ones(2, 1, 2)},
        None,
    )
    model = RAAVFusionModel(
        {"frame": 2, "context": 2},
        2,
        1,
        ("frame",),
        ("context",),
    ).eval()

    result = explain_batch(model, batch, modalities=("context",), ablation_policy="remove")

    assert result["samples"][0]["modalities"]["context"]["status"] == "evaluated"


def test_tomar_requires_the_anchor_for_remove_and_mask() -> None:
    batch = Batch(
        {"anchor": torch.ones(2, 1, 2), "extra": torch.ones(2, 1, 2)},
        None,
    )
    model = TOMARFusionModel({"anchor": 2, "extra": 2}, 2, "anchor", 1).eval()

    for policy in ("remove", "mask"):
        result = explain_batch(model, batch, modalities=("anchor",), ablation_policy=policy)
        assert result["samples"][0]["modalities"]["anchor"]["status"] == "unsupported"


def test_tomar_allows_additional_modality_intervention() -> None:
    batch = Batch(
        {"anchor": torch.ones(2, 1, 2), "extra": torch.ones(2, 1, 2)},
        None,
    )
    model = TOMARFusionModel({"anchor": 2, "extra": 2}, 2, "anchor", 1).eval()

    result = explain_batch(model, batch, modalities=("extra",), ablation_policy="mask")

    assert result["samples"][0]["modalities"]["extra"]["status"] == "evaluated"
