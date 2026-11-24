import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.tfs_fusion import TFSDCMMOEFusionModel, tfs_dcmmmoe_fusion_model


def test_tfs_dcmmmoe_learns_pair_weights_and_masks_unavailable_pairs():
    model = TFSDCMMOEFusionModel(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4), "video": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Linear(4, 2),
        gate_hidden=5,
    )
    batch = Batch(
        inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 3)},
        targets=None,
        masks={"text_mask": torch.tensor([1, 1]), "audio_mask": torch.tensor([1, 0])},
    )

    out = model(batch)

    assert out.preds.shape == (2, 2)
    assert out.aux is not None
    weights = out.aux["pair_weights"]
    assert weights.shape == (2, 2)
    assert torch.allclose(weights[0].sum(0), torch.ones(()))
    assert torch.allclose(weights[1], torch.zeros(2))


def test_tfs_dcmmoe_requires_two_available_modalities():
    model = TFSDCMMOEFusionModel(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4)}, shared_dim=4, head=nn.Identity()
    )

    with pytest.raises(ValueError, match="at least two available modalities"):
        model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))


def test_tfs_dcmmoe_factory_is_registered():
    assert MODELS.get("tfs_dcmmmoe_fusion_model") is tfs_dcmmmoe_fusion_model
    model = tfs_dcmmmoe_fusion_model(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4)}, shared_dim=4, head=nn.Identity()
    )
    assert isinstance(model, TFSDCMMOEFusionModel)
