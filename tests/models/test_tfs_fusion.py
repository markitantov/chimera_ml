import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.tfs_fusion import TFSFusionModel, tfs_fusion_model


def _batch() -> Batch:
    return Batch(
        inputs={
            "text": torch.randn(2, 3, 4),
            "audio": torch.randn(2, 3, 5),
            "video": torch.randn(2, 3, 6),
        },
        targets=None,
        masks={
            "text_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "audio_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "video_mask": torch.tensor([[1, 0, 0], [1, 1, 1]]),
        },
    )


def test_tfs_fuses_directed_pairs_and_returns_auxiliary_features():
    model = TFSFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=4,
        input_dims={"audio": 5, "video": 6},
        head=nn.Linear(4, 2),
    )

    out = model(_batch())

    assert out.preds.shape == (2, 2)
    assert out.aux is not None
    assert out.aux["fused"].shape == (2, 4)
    assert "pair_text->audio" in out.aux
    assert out.aux["pair_text->audio"].shape == (2, 4)


def test_tfs_supports_explicit_groups_and_missing_modalities():
    model = TFSFusionModel(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4), "video": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Identity(),
        groups=((("text", "audio"), ("audio", "text")), (("video", "text"),)),
    )

    out = Batch(inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 3)}, targets=None)
    result = model(out)

    assert result.preds.shape == (2, 4)
    assert "pair_video->text" not in result.aux


def test_tfs_requires_two_available_modalities():
    model = TFSFusionModel(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Identity(),
    )

    with pytest.raises(ValueError, match="at least two available modalities"):
        model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))


def test_tfs_factory_is_registered():
    assert MODELS.get("tfs_fusion_model") is tfs_fusion_model
    model = tfs_fusion_model(
        encoders={"text": nn.Linear(3, 4), "audio": nn.Linear(3, 4)}, shared_dim=4, head=nn.Identity()
    )
    assert isinstance(model, TFSFusionModel)
