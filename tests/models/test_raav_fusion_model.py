import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.raav_fusion_model import RAAVFusionModel, raav_fusion_model


def make_batch() -> Batch:
    return Batch(
        inputs={
            "face": torch.randn(2, 3, 4),
            "behavior": torch.randn(2, 3, 5),
            "audio": torch.randn(2, 2, 6),
            "text": torch.randn(2, 4, 7),
        },
        targets=None,
        masks={
            "face_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "behavior_mask": torch.tensor([[1, 0, 1], [0, 0, 0]]),
            "audio_mask": torch.tensor([[1, 1], [1, 0]]),
            "text_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]),
        },
    )


def make_model() -> RAAVFusionModel:
    return RAAVFusionModel(
        input_dims={"face": 4, "behavior": 5, "audio": 6, "text": 7},
        shared_dim=4,
        output_dim=2,
        frame_modalities=("face", "behavior"),
        context_modalities=("audio", "text"),
    ).eval()


def test_raav_fuses_frames_and_all_context_modalities():
    model = make_model()
    output = model(make_batch())

    assert output.preds.shape == (2, 3, 2)
    assert output.aux is not None
    assert output.aux["frame_weights"].shape == (2, 2, 3)
    assert output.aux["context_delta"].shape == (2, 3, 4)
    assert output.aux["context_attention_audio"].shape == (2, 3, 1)
    assert output.aux["context_attention_text"].shape == (2, 3, 1)
    assert torch.allclose(output.aux["frame_weights"].sum(dim=1), torch.ones(2, 3))
    assert torch.isfinite(output.preds).all()


def test_raav_assigns_one_to_the_only_available_frame_modality():
    model = make_model()
    batch = Batch(
        inputs={"face": torch.randn(2, 3, 4), "audio": torch.randn(2, 2, 6)},
        targets=None,
        masks={"face_mask": torch.ones(2, 3), "audio_mask": torch.ones(2, 2)},
    )

    output = model(batch)

    assert torch.equal(output.aux["frame_weights"], torch.ones(2, 1, 3))


def test_raav_allows_missing_context_modalities():
    model = make_model()
    batch = Batch(
        inputs={"face": torch.randn(2, 3, 4), "behavior": torch.randn(2, 3, 5)},
        targets=None,
    )

    output = model(batch)

    assert output.preds.shape == (2, 3, 2)
    assert torch.equal(output.aux["context_delta"], torch.zeros(2, 3, 4))
    assert torch.isfinite(output.preds).all()


def test_raav_uses_regular_multihead_attention():
    model = make_model()

    assert all(isinstance(block, nn.MultiheadAttention) for block in model.context_attention.values())


def test_raav_validates_frame_lengths_and_available_inputs():
    model = make_model()

    with pytest.raises(ValueError, match="same temporal length"):
        model(
            Batch(
                inputs={"face": torch.randn(2, 3, 4), "behavior": torch.randn(2, 2, 5)},
                targets=None,
            )
        )

    with pytest.raises(ValueError, match="no available frame modalities"):
        model(Batch(inputs={"audio": torch.randn(2, 2, 6)}, targets=None))


def test_raav_factory_is_registered():
    assert MODELS.get("raav_fusion_model") is raav_fusion_model
    assert isinstance(
        raav_fusion_model(
            input_dims={"frame": 3, "context": 4},
            shared_dim=4,
            output_dim=2,
            frame_modalities=("frame",),
            context_modalities=("context",),
        ),
        RAAVFusionModel,
    )
