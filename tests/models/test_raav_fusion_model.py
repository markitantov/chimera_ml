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
    assert output.aux["context_attention_audio"].shape == (2, 3, 4)
    assert output.aux["context_compression_attention_audio"].shape == (2, 4, 2)
    assert output.aux["context_attention_text"].shape == (2, 3, 4)
    assert output.aux["context_compression_attention_text"].shape == (2, 4, 4)
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
    assert all(isinstance(block, nn.MultiheadAttention) for block in model.context_compression.values())
    assert model.context_latents["audio"].shape == (1, 4, 4)


def test_raav_validates_context_latent_count_and_handles_masked_context():
    with pytest.raises(ValueError, match="num_context_latents must be positive"):
        RAAVFusionModel(
            {"frame": 2, "context": 2},
            4,
            1,
            ("frame",),
            ("context",),
            num_context_latents=0,
        )

    model = make_model()
    output = model(
        Batch(
            inputs={
                "face": torch.randn(2, 3, 4),
                "behavior": torch.randn(2, 3, 5),
                "audio": torch.randn(2, 2, 6),
                "text": torch.randn(2, 4, 7),
            },
            targets=None,
            masks={
                "face_mask": torch.ones(2, 3),
                "behavior_mask": torch.ones(2, 3),
                "audio_mask": torch.zeros(2, 2),
                "text_mask": torch.zeros(2, 4),
            },
        )
    )

    assert torch.isfinite(output.preds).all()
    assert torch.equal(output.aux["context_delta"], torch.zeros_like(output.aux["context_delta"]))
    assert torch.equal(
        output.aux["context_compression_attention_audio"],
        torch.zeros_like(output.aux["context_compression_attention_audio"]),
    )


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


def configure_identity_attention(block: nn.MultiheadAttention):
    with torch.no_grad():
        identity = torch.eye(block.embed_dim)
        block.in_proj_weight.copy_(torch.cat((identity, identity, identity)))
        block.in_proj_bias.zero_()
        block.out_proj.weight.copy_(identity)
        block.out_proj.bias.zero_()


def test_raav_frame_attention_depends_on_the_frame_query():
    model = RAAVFusionModel(
        {"frame": 2, "context": 2},
        2,
        1,
        ("frame",),
        ("context",),
        num_context_latents=2,
        dropout=0.0,
        num_heads=1,
    ).eval()
    configure_identity_attention(model.context_compression["context"])
    configure_identity_attention(model.context_attention["context"])
    with torch.no_grad():
        model.proj["frame"].weight.copy_(torch.eye(2))
        model.proj["frame"].bias.zero_()
        model.proj["context"].weight.copy_(torch.eye(2))
        model.proj["context"].bias.zero_()
        model.context_latents["context"].copy_(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))

    output = model(
        Batch(
            inputs={
                "frame": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                "context": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            },
            targets=None,
        )
    )

    attention = output.aux["context_attention_context"]
    assert not torch.allclose(attention[:, 0], attention[:, 1])


def test_raav_sums_independent_context_corrections_before_normalization():
    model = make_model()
    inputs = {
        "face": torch.ones(1, 3, 4),
        "behavior": torch.ones(1, 3, 5),
        "audio": torch.ones(1, 2, 6),
        "text": torch.ones(1, 2, 7),
    }
    both = model(Batch(inputs=inputs, targets=None)).aux["context_delta"]
    audio_batch = Batch(
        inputs={key: value for key, value in inputs.items() if key != "text"},
        targets=None,
    )
    text_batch = Batch(
        inputs={key: value for key, value in inputs.items() if key != "audio"},
        targets=None,
    )
    audio = model(audio_batch).aux["context_delta"]
    text = model(text_batch).aux["context_delta"]

    assert torch.allclose(both, audio + text)
