import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.tomar_fusion import TOMARFusionModel, tomar_fusion_model


def test_tomar_refines_anchor_with_available_modalities():
    model = TOMARFusionModel(
        input_dims={"text": 4, "audio": 4, "video": 4},
        shared_dim=4,
        output_dim=2,
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
        input_dims={"text": 3, "audio": 3},
        shared_dim=4,
        output_dim=4,
        anchor_modality="text",
    )

    out = model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))
    assert out.preds.shape == (2, 4)
    assert "delta_audio" not in out.aux

    with pytest.raises(ValueError, match="Anchor modality 'text' is missing"):
        model(Batch(inputs={"audio": torch.randn(2, 3)}, targets=None))


def test_tomar_validates_anchor_and_registers_factory():
    with pytest.raises(ValueError, match="Unknown anchor modality"):
        TOMARFusionModel(input_dims={"text": 3}, shared_dim=4, output_dim=4, anchor_modality="audio")

    assert MODELS.get("tomar_fusion_model") is tomar_fusion_model
    model = tomar_fusion_model(input_dims={"text": 3}, shared_dim=4, output_dim=4, anchor_modality="text")
    assert isinstance(model, TOMARFusionModel)


def _tomar_joint_model() -> TOMARFusionModel:
    return TOMARFusionModel(
        input_dims={"text": 2, "audio": 2, "video": 2},
        shared_dim=2,
        output_dim=2,
        anchor_modality="text",
        gate_hidden=3,
    ).eval()


def test_tomar_gate_depends_on_anchor_and_additional_representation():
    model = _tomar_joint_model()
    fixed_audio = torch.tensor([[[0.5, 1.0]]])
    first = Batch(
        inputs={"text": torch.tensor([[[1.0, 0.0]]]), "audio": fixed_audio},
        targets=None,
        masks={"text_mask": torch.ones(1), "audio_mask": torch.ones(1)},
    )
    second = Batch(
        inputs={"text": torch.tensor([[[0.0, 1.0]]]), "audio": fixed_audio},
        targets=None,
        masks={"text_mask": torch.ones(1), "audio_mask": torch.ones(1)},
    )

    first_rho = model(first).aux["rho_audio"]
    second_rho = model(second).aux["rho_audio"]

    assert not torch.allclose(first_rho, second_rho)


def test_tomar_residual_matches_manual_reference_and_masks_coefficients():
    model = TOMARFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        anchor_modality="text",
        gate_hidden=2,
    ).eval()
    with torch.no_grad():
        for projection in model.proj.values():
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        model.delta["audio"].weight.copy_(torch.eye(2))
        model.delta["audio"].bias.zero_()
        model.rho["audio"][0].weight.zero_()
        model.rho["audio"][0].bias.zero_()
        model.rho["audio"][2].weight.zero_()
        model.rho["audio"][2].bias.zero_()

    batch = Batch(
        inputs={
            "text": torch.tensor([[[1.0, 3.0]], [[2.0, 2.0]]]),
            "audio": torch.tensor([[[4.0, 1.0]], [[100.0, 100.0]]]),
        },
        targets=None,
        masks={"text_mask": torch.ones(2), "audio_mask": torch.tensor([1, 0])},
    )

    output = model(batch)
    expected_first = model.norm(torch.tensor([[1.0, 3.0]]) + 0.5 * torch.tensor([[4.0, 1.0]]))
    expected_second = model.norm(torch.tensor([[2.0, 2.0]]))

    assert torch.allclose(output.aux["fused"][0], expected_first)
    assert torch.allclose(output.aux["fused"][1], expected_second)
    assert torch.allclose(output.aux["rho_audio"][1], torch.zeros(1))
    assert torch.allclose(output.aux["delta_audio"][1], torch.zeros(2))


def test_tomar_rejects_per_sample_missing_anchor():
    model = _tomar_joint_model()
    batch = Batch(
        inputs={"text": torch.ones(2, 1, 2), "audio": torch.ones(2, 1, 2)},
        targets=None,
        masks={"text_mask": torch.tensor([1, 0]), "audio_mask": torch.ones(2)},
    )

    with pytest.raises(ValueError, match="unavailable for one or more samples"):
        model(batch)


def test_tomar_gates_and_delta_networks_receive_finite_gradients():
    model = TOMARFusionModel(
        input_dims={"text": 3, "audio": 3, "video": 3},
        shared_dim=4,
        output_dim=2,
        anchor_modality="text",
        gate_hidden=5,
    )
    batch = Batch(
        inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 3), "video": torch.randn(2, 3)},
        targets=None,
    )

    model(batch).preds.sum().backward()

    residual_parameters = list(model.rho.parameters()) + list(model.delta.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in residual_parameters)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.proj.parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.prediction.parameters()
    )


def test_tomar_state_round_trip_preserves_predictions():
    torch.manual_seed(13)
    model = TOMARFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        anchor_modality="text",
    ).eval()
    restored = TOMARFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        anchor_modality="text",
    ).eval()
    restored.load_state_dict(model.state_dict())
    batch = Batch(
        inputs={"text": torch.randn(2, 3, 2), "audio": torch.randn(2, 2)},
        targets=None,
    )

    assert torch.equal(model(batch).preds, restored(batch).preds)


def test_tomar_empty_additional_modality_has_zero_residual_contribution():
    model = TOMARFusionModel(
        input_dims={"text": 3, "audio": 4},
        shared_dim=4,
        anchor_modality="text",
        output_dim=2,
    ).eval()
    batch = Batch(
        inputs={"text": torch.randn(2, 1, 3), "audio": torch.empty(2, 0, 4)},
        targets=None,
        masks={
            "text_temporal_mask": torch.ones(2, 1, dtype=torch.bool),
            "audio_temporal_mask": torch.zeros(2, 0, dtype=torch.bool),
        },
    )

    output = model(batch)

    assert output.aux is not None
    assert torch.isfinite(output.preds).all()
    assert torch.equal(output.aux["rho_audio"], torch.zeros_like(output.aux["rho_audio"]))
    assert torch.equal(output.aux["delta_audio"], torch.zeros_like(output.aux["delta_audio"]))


def test_tomar_empty_anchor_preserves_explicit_error():
    model = TOMARFusionModel(
        input_dims={"text": 3, "audio": 4},
        shared_dim=4,
        anchor_modality="text",
        output_dim=2,
    ).eval()
    batch = Batch(
        inputs={"text": torch.empty(2, 0, 3), "audio": torch.randn(2, 1, 4)},
        targets=None,
        masks={
            "text_temporal_mask": torch.zeros(2, 0, dtype=torch.bool),
            "audio_temporal_mask": torch.ones(2, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match="Anchor modality 'text' is unavailable"):
        model(batch)
