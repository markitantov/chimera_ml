import pytest
import torch
import torch.nn as nn

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.raav_fusion import RAAVFusionModel, raav_fusion_model


def test_raav_fuses_frames_and_refines_with_context_latents():
    model = RAAVFusionModel(
        input_dims={"audio": 4, "video": 4, "context": 4},
        shared_dim=4,
        output_dim=2,
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
        input_dims={"audio": 3, "video": 3, "context": 3},
        shared_dim=4,
        output_dim=4,
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
        input_dims={"audio": 3, "context": 3},
        shared_dim=4,
        output_dim=4,
        frame_modalities=("audio",),
        context_modality="context",
    )

    with pytest.raises(ValueError, match="no available frame modalities"):
        model(Batch(inputs={"context": torch.randn(2, 3)}, targets=None))

    assert MODELS.get("raav_fusion_model") is raav_fusion_model
    assert isinstance(
        raav_fusion_model(
            input_dims={"audio": 3, "context": 3},
            shared_dim=4,
            output_dim=4,
            frame_modalities=("audio",),
            context_modality="context",
        ),
        RAAVFusionModel,
    )


def _raav_frame_model() -> RAAVFusionModel:
    model = RAAVFusionModel(
        input_dims={"frame": 4, "context": 4},
        shared_dim=4,
        output_dim=4,
        frame_modalities=("frame",),
        context_modality="context",
        num_latents=2,
    ).eval()
    with torch.no_grad():
        model.gate[0].weight.zero_()
        model.gate[0].bias.zero_()
        model.gate[2].weight.zero_()
        model.gate[2].bias.zero_()
        model.refine.weight.copy_(torch.eye(4))
        model.refine.bias.zero_()
    return model


def test_raav_context_refinement_is_frame_dependent():
    model = _raav_frame_model()
    frames = Batch(
        inputs={"frame": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])},
        targets=None,
        masks={"frame_mask": torch.ones(1)},
    )
    with_context = Batch(
        inputs={
            "frame": frames.inputs["frame"],
            "context": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        },
        targets=None,
        masks={"frame_mask": torch.ones(1), "context_mask": torch.ones(1)},
    )

    delta = model(with_context).preds - model(frames).preds

    assert not torch.allclose(delta[:, 0], delta[:, 1])


def test_raav_invalid_frame_outputs_are_reapplied_after_context_refinement():
    model = _raav_frame_model()
    batch = Batch(
        inputs={
            "frame": torch.ones(1, 3, 4),
            "context": torch.ones(1, 2, 4),
        },
        targets=None,
        masks={"frame_mask": torch.tensor([[1, 1, 0]]), "context_mask": torch.ones(1)},
    )

    output = model(batch)

    assert torch.allclose(output.preds[:, 2], torch.zeros_like(output.preds[:, 2]))
    assert torch.equal(output.aux["frame_mask"], torch.tensor([[True, True, False]]))


def test_raav_fully_masked_context_has_no_effect():
    model = _raav_frame_model()
    frames = Batch(
        inputs={"frame": torch.ones(1, 2, 4)},
        targets=None,
        masks={"frame_mask": torch.ones(1)},
    )
    masked_context = Batch(
        inputs={"frame": frames.inputs["frame"], "context": torch.full((1, 3, 4), 100.0)},
        targets=None,
        masks={"frame_mask": torch.ones(1), "context_mask": torch.zeros(1, 3)},
    )

    assert torch.allclose(model(frames).preds, model(masked_context).preds)


def test_raav_cross_attention_and_latents_receive_finite_gradients():
    model = RAAVFusionModel(
        input_dims={"frame": 3, "context": 3},
        shared_dim=4,
        output_dim=2,
        frame_modalities=("frame",),
        context_modality="context",
        num_latents=2,
    )
    batch = Batch(
        inputs={"frame": torch.randn(2, 3, 3), "context": torch.randn(2, 4, 3)},
        targets=None,
    )

    model(batch).preds.sum().backward()

    assert model.latents.grad is not None and torch.isfinite(model.latents.grad).all()
    attention_parameters = list(model.context_latent_attn.parameters()) + list(model.frame_to_context_attn.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in attention_parameters
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.proj.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.prediction.parameters()
    )


def test_raav_state_round_trip_preserves_predictions():
    torch.manual_seed(11)
    model = RAAVFusionModel(
        input_dims={"frame": 4, "context": 4},
        shared_dim=4,
        output_dim=4,
        frame_modalities=("frame",),
        context_modality="context",
        num_latents=2,
    ).eval()
    restored = RAAVFusionModel(
        input_dims={"frame": 4, "context": 4},
        shared_dim=4,
        output_dim=4,
        frame_modalities=("frame",),
        context_modality="context",
        num_latents=2,
    ).eval()
    restored.load_state_dict(model.state_dict())
    batch = Batch(
        inputs={"frame": torch.randn(2, 2, 4), "context": torch.randn(2, 3, 4)},
        targets=None,
    )

    assert torch.equal(model(batch).preds, restored(batch).preds)


def test_raav_temporal_encoder_does_not_attend_to_padding():
    def build() -> RAAVFusionModel:
        temporal_layer = nn.TransformerEncoderLayer(d_model=4, nhead=1, dropout=0.0, batch_first=True)
        return RAAVFusionModel(
            input_dims={"frame": 4, "context": 4},
            shared_dim=4,
            output_dim=4,
            frame_modalities=("frame",),
            context_modality="context",
            temporal=nn.TransformerEncoder(temporal_layer, num_layers=1),
        ).eval()

    short_model = build()
    padded_model = build()
    padded_model.load_state_dict(short_model.state_dict())
    short = Batch(
        inputs={"frame": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])},
        targets=None,
        masks={"frame_mask": torch.ones(1, 2)},
    )
    padded = Batch(
        inputs={"frame": torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [99.0, 99.0, 99.0, 99.0]]])},
        targets=None,
        masks={"frame_mask": torch.tensor([[1, 1, 0]])},
    )

    assert torch.allclose(short_model(short).preds, padded_model(padded).preds[:, :2], atol=1e-6)


def test_raav_missing_context_has_no_learned_context_residual():
    model = _raav_frame_model()
    with torch.no_grad():
        model.refine.bias.fill_(3.0)
    batch = Batch(
        inputs={"frame": torch.ones(1, 2, 4)},
        targets=None,
        masks={"frame_mask": torch.ones(1)},
    )

    output = model(batch)

    assert torch.allclose(output.aux["context_delta"], torch.zeros_like(output.aux["context_delta"]))
