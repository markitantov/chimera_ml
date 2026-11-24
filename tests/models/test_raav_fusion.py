import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.raav_fusion import RAAVFusionModel, raav_fusion_model


def test_raav_fuses_frames_and_refines_with_context_latents():
    model = RAAVFusionModel(
        encoders={"audio": nn.Identity(), "video": nn.Identity(), "context": nn.Identity()},
        shared_dim=4,
        head=nn.Linear(4, 2),
        frame_modalities=("audio", "video"),
        context_modality="context",
        num_latents=3,
    )
    batch = Batch(
        inputs={
            "audio": torch.randn(2, 3, 4),
            "video": torch.randn(2, 3, 4),
            "context": torch.randn(2, 2, 4),
        },
        targets=None,
        masks={
            "audio_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "video_mask": torch.tensor([[1, 0, 1], [0, 0, 0]]),
            "context_mask": torch.tensor([[1, 1], [1, 0]]),
        },
    )

    out = model(batch)

    assert out.preds.shape == (2, 3, 2)
    assert out.aux is not None
    assert out.aux["frame_weights"].shape == (2, 2, 3)
    assert out.aux["context"].shape == (2, 3, 4)
    assert torch.allclose(out.aux["frame_weights"][0].sum(0), torch.ones(3))
    assert torch.allclose(out.aux["frame_weights"][1, 1], torch.zeros(3))


def test_raav_allows_missing_context_and_validates_frame_lengths():
    model = RAAVFusionModel(
        encoders={"audio": nn.Linear(3, 4), "video": nn.Linear(3, 4), "context": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Identity(),
        frame_modalities=("audio", "video"),
        context_modality="context",
    )

    missing_context = model(
        Batch(
            inputs={"audio": torch.randn(2, 2, 3), "video": torch.randn(2, 2, 3)},
            targets=None,
        )
    )
    assert missing_context.preds.shape == (2, 2, 4)
    assert not missing_context.aux["context_mask"].any()

    with pytest.raises(ValueError, match="same temporal length"):
        model(
            Batch(
                inputs={"audio": torch.randn(2, 2, 3), "video": torch.randn(2, 3, 3)},
                targets=None,
            )
        )


def test_raav_requires_an_available_frame_and_has_registered_factory():
    model = RAAVFusionModel(
        encoders={"audio": nn.Linear(3, 4), "context": nn.Linear(3, 4)},
        shared_dim=4,
        head=nn.Identity(),
        frame_modalities=("audio",),
        context_modality="context",
    )

    with pytest.raises(ValueError, match="no available frame modalities"):
        model(Batch(inputs={"context": torch.randn(2, 3)}, targets=None))

    assert MODELS.get("raav_fusion_model") is raav_fusion_model
    assert isinstance(
        raav_fusion_model(
            encoders={"audio": nn.Linear(3, 4), "context": nn.Linear(3, 4)},
            shared_dim=4,
            head=nn.Identity(),
            frame_modalities=("audio",),
            context_modality="context",
        ),
        RAAVFusionModel,
    )
