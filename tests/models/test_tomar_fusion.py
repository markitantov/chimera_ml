import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.tomar_fusion import TOMARFusionModel, tomar_fusion_model


def test_tomar_refines_anchor_with_available_modalities():
    model = TOMARFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=4,
        head=nn.Linear(4, 2),
        anchor_modality="text",
        gate_hidden=5,
    )
    batch = Batch(
        inputs={
            "text": torch.randn(2, 3, 4),
            "audio": torch.randn(2, 3, 4),
            "video": torch.randn(2, 4),
        },
        targets=None,
        masks={"text_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]), "audio_mask": torch.tensor([[1, 0, 0], [0, 0, 0]])},
    )

    out = model(batch)

    assert out.preds.shape == (2, 2)
    assert out.aux is not None
    assert out.aux["anchor"].shape == (2, 4)
    assert out.aux["fused"].shape == (2, 4)
    assert out.aux["rho_audio"].shape == (2, 1)
    assert torch.allclose(out.aux["delta_audio"][1], torch.zeros(4))
    assert "rho_video" in out.aux


def test_tomar_skips_missing_extras_and_requires_anchor():
    model = TOMARFusionModel(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Identity(),
        anchor_modality="text",
    )

    out = model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))
    assert out.preds.shape == (2, 4)
    assert "delta_audio" not in out.aux

    with pytest.raises(ValueError, match="Anchor modality 'text' is missing"):
        model(Batch(inputs={"audio": torch.randn(2, 3)}, targets=None))


def test_tomar_validates_anchor_and_registers_factory():
    with pytest.raises(ValueError, match="Unknown anchor modality"):
        TOMARFusionModel(encoders={"text": nn.Linear(3, 4)}, shared_dim=4, head=nn.Identity(), anchor_modality="audio")

    assert MODELS.get("tomar_fusion_model") is tomar_fusion_model
    model = tomar_fusion_model(
        encoders={"text": nn.Linear(3, 4)}, shared_dim=4, head=nn.Identity(), anchor_modality="text"
    )
    assert isinstance(model, TOMARFusionModel)
