import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.core.registry import MODELS
from chimera_ml.models.tfs_fusion import TFSDCMMOEFusionModel, tfs_dcmmmoe_fusion_model


def test_tfs_dcmmmoe_learns_pair_weights_and_masks_unavailable_pairs():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 3, "audio": 3, "video": 3},
        shared_dim=4,
        output_dim=2,
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
    model = TFSDCMMOEFusionModel(input_dims={"text": 3, "audio": 3}, shared_dim=4, output_dim=4)

    with pytest.raises(ValueError, match="at least two available modalities"):
        model(Batch(inputs={"text": torch.randn(2, 3)}, targets=None))


def test_tfs_dcmmoe_factory_is_registered():
    assert MODELS.get("tfs_dcmmmoe_fusion_model") is tfs_dcmmmoe_fusion_model
    model = tfs_dcmmmoe_fusion_model(input_dims={"text": 3, "audio": 3}, shared_dim=4, output_dim=4)
    assert isinstance(model, TFSDCMMOEFusionModel)


def _dcmmmoe_batch() -> Batch:
    return Batch(
        inputs={
            "text": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            "audio": torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]]),
            "video": torch.tensor([[[1.0, 1.0]], [[2.0, 0.0]]]),
        },
        targets=None,
        masks={"text_mask": torch.ones(2), "audio_mask": torch.ones(2), "video_mask": torch.ones(2)},
    )


def _set_identity_output(model: TFSDCMMOEFusionModel) -> None:
    with torch.no_grad():
        model.output.weight.copy_(torch.eye(2))
        model.output.bias.zero_()


def test_tfs_dcmmmoe_groups_change_the_computation():
    first = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2, "video": 2},
        shared_dim=2,
        output_dim=2,
        groups=((("text", "audio"), ("audio", "text")),),
        gate_hidden=3,
    )
    second = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2, "video": 2},
        shared_dim=2,
        output_dim=2,
        groups=((("text", "video"), ("video", "text")),),
        gate_hidden=3,
    )
    second.load_state_dict(first.state_dict())
    _set_identity_output(first)
    _set_identity_output(second)

    assert not torch.allclose(first(_dcmmmoe_batch()).aux["fused"], second(_dcmmmoe_batch()).aux["fused"])


def test_tfs_dcmmmoe_pair_weights_are_input_dependent_and_normalized():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        gate_hidden=3,
    )
    with torch.no_grad():
        for projection in model.proj.values():
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        model.gates[0].weight.zero_()
        model.gates[0].bias.zero_()
        model.gates[0].weight[0, 0] = 1.0
        model.gates[2].weight.fill_(1.0)
        model.gates[2].bias.zero_()

    first = Batch(
        inputs={"text": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), "audio": torch.tensor([[[2.0, 0.0]]])},
        targets=None,
        masks={"text_mask": torch.ones(1), "audio_mask": torch.ones(1)},
    )
    second = Batch(
        inputs={"text": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), "audio": torch.tensor([[[0.0, 2.0]]])},
        targets=None,
        masks={"text_mask": torch.ones(1), "audio_mask": torch.ones(1)},
    )

    first_weights = model(first).aux["pair_weights"]
    second_weights = model(second).aux["pair_weights"]

    assert not torch.allclose(first_weights, second_weights)
    assert torch.allclose(first_weights.sum(1), torch.ones(1))
    assert torch.isfinite(second_weights).all()


def test_tfs_dcmmmoe_unavailable_pairs_have_zero_weight_and_finite_fused_output():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2, "video": 2},
        shared_dim=2,
        output_dim=2,
        gate_hidden=3,
    )
    batch = _dcmmmoe_batch()
    batch.masks["audio_mask"][1] = 0
    batch.masks["video_mask"][1] = 0

    output = model(batch)

    assert torch.equal(output.aux["pair_weights"][1], torch.zeros(6))
    assert torch.allclose(output.aux["fused"][1], torch.zeros(2))
    assert torch.isfinite(output.preds).all()


def test_tfs_dcmmmoe_backpropagates_through_gate_and_pairwise_fusion():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 3, "audio": 3, "video": 3},
        shared_dim=4,
        output_dim=2,
        gate_hidden=5,
    )
    batch = Batch(
        inputs={"text": torch.randn(2, 3), "audio": torch.randn(2, 3), "video": torch.randn(2, 3)},
        targets=None,
    )

    model(batch).preds.sum().backward()

    gate_grads = [parameter.grad for parameter in model.gates.parameters()]
    backbone_grads = [parameter.grad for parameter in model.triple_fusion.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gate_grads)
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in backbone_grads)


def test_tfs_dcmmmoe_state_round_trip_preserves_predictions():
    torch.manual_seed(7)
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        gate_hidden=3,
    ).eval()
    restored = TFSDCMMOEFusionModel(
        input_dims={"text": 2, "audio": 2},
        shared_dim=2,
        output_dim=2,
        gate_hidden=3,
    ).eval()
    restored.load_state_dict(model.state_dict())

    assert torch.equal(model(_dcmmmoe_batch()).preds, restored(_dcmmmoe_batch()).preds)


def test_tfs_dcmmmoe_zero_length_modality_has_zero_pair_outputs_and_weights():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 3, "audio": 4, "video": 5}, shared_dim=4, output_dim=2, gate_hidden=3
    ).eval()
    batch = Batch(
        inputs={
            "text": torch.randn(2, 2, 3),
            "audio": torch.randn(2, 1, 4),
            "video": torch.empty(2, 0, 5),
        },
        targets=None,
        masks={
            "text_temporal_mask": torch.ones(2, 2, dtype=torch.bool),
            "audio_temporal_mask": torch.ones(2, 1, dtype=torch.bool),
            "video_temporal_mask": torch.zeros(2, 0, dtype=torch.bool),
        },
    )

    output = model(batch)

    assert output.aux is not None
    assert torch.isfinite(output.preds).all()
    for key, value in output.aux.items():
        if key.startswith("pair_valid_") and "video" in key:
            assert not value.any()
        if key.startswith("pair_") and not key.startswith("pair_valid_") and "video" in key:
            assert torch.equal(value, torch.zeros_like(value))
    pair_weights = output.aux["pair_weights"]
    assert torch.equal(pair_weights[:, [1, 2, 4, 5]], torch.zeros_like(pair_weights[:, [1, 2, 4, 5]]))


def test_tfs_dcmmmoe_all_invalid_nonempty_modalities_keep_zero_fused_representation():
    model = TFSDCMMOEFusionModel(
        input_dims={"text": 3, "audio": 4, "video": 5}, shared_dim=4, output_dim=2, gate_hidden=3
    ).eval()
    batch = Batch(
        inputs={
            "text": torch.randn(2, 1, 3),
            "audio": torch.randn(2, 1, 4),
            "video": torch.randn(2, 1, 5),
        },
        targets=None,
        masks={
            "text_temporal_mask": torch.zeros(2, 1, dtype=torch.bool),
            "audio_temporal_mask": torch.zeros(2, 1, dtype=torch.bool),
            "video_temporal_mask": torch.zeros(2, 1, dtype=torch.bool),
        },
    )

    output = model(batch)

    assert output.aux is not None
    assert torch.equal(output.aux["fused"], torch.zeros_like(output.aux["fused"]))
    assert torch.equal(output.aux["pair_weights"], torch.zeros_like(output.aux["pair_weights"]))
    assert torch.isfinite(output.preds).all()
