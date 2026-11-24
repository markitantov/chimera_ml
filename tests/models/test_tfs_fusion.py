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


def _identity_tfs() -> TFSFusionModel:
    model = TFSFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=2,
        head=nn.Identity(),
    )
    with torch.no_grad():
        for layer in model.group_layers:
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        model.output.weight.copy_(torch.eye(2))
        model.output.bias.zero_()
    model.eval()
    return model


def test_tfs_uses_only_pairs_available_for_each_sample():
    model = _identity_tfs()
    batch = Batch(
        inputs={
            "text": torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            "audio": torch.tensor([[[0.0, 1.0]], [[9.0, 9.0]]]),
            "video": torch.tensor([[[7.0, 7.0]], [[1.0, 0.0]]]),
        },
        targets=None,
        masks={
            "text_mask": torch.tensor([1, 1]),
            "audio_mask": torch.tensor([1, 0]),
            "video_mask": torch.tensor([0, 1]),
        },
    )

    output = model(batch)

    assert torch.isfinite(output.aux["fused"]).all()
    assert torch.equal(output.aux["pair_valid_text->video"], torch.tensor([False, True]))
    assert torch.equal(output.aux["pair_valid_text->audio"], torch.tensor([True, False]))
    assert torch.equal(output.aux["pair_weights"][0, [2, 5]], torch.zeros(2))
    assert torch.equal(output.aux["pair_weights"][1, [0, 3]], torch.zeros(2))


def test_tfs_has_explicit_zero_policy_for_samples_without_valid_pairs():
    model = _identity_tfs()
    batch = Batch(
        inputs={"text": torch.ones(2, 1, 2), "audio": torch.ones(2, 1, 2)},
        targets=None,
        masks={"text_mask": torch.tensor([1, 1]), "audio_mask": torch.tensor([1, 0])},
    )

    output = model(batch)

    assert torch.allclose(output.aux["fused"][1], torch.zeros(2))
    assert torch.isfinite(output.preds).all()


def test_tfs_directed_pair_outputs_are_not_forced_to_be_symmetric():
    model = _identity_tfs()
    left = (torch.tensor([[[2.0, 0.0], [0.0, 0.0]]]), torch.tensor([[1, 0]]))
    right = (torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), torch.tensor([[1, 1]]))

    left_to_right, _ = model._pair(left, right)
    right_to_left, _ = model._pair(right, left)

    assert not torch.allclose(left_to_right, right_to_left)


def test_tfs_group_configuration_changes_hierarchical_aggregation():
    inputs = {
        "text": torch.tensor([[[1.0, 0.0]]]),
        "audio": torch.tensor([[[0.0, 1.0]]]),
        "video": torch.tensor([[[2.0, 0.0]]]),
    }
    masks = {"text_mask": torch.ones(1), "audio_mask": torch.ones(1), "video_mask": torch.ones(1)}
    batch = Batch(inputs=inputs, targets=None, masks=masks)

    all_pairs = _identity_tfs()
    split_pairs = TFSFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=2,
        head=nn.Identity(),
        groups=((("text", "audio"),), (("audio", "text"), ("text", "video"), ("video", "text"))),
    )
    with torch.no_grad():
        for layer in split_pairs.group_layers:
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
        split_pairs.output.weight.copy_(torch.eye(2))
        split_pairs.output.bias.zero_()
    split_pairs.eval()

    assert not torch.allclose(all_pairs(batch).aux["fused"], split_pairs(batch).aux["fused"])


def test_tfs_state_round_trip_preserves_predictions():
    model = _identity_tfs()
    batch = Batch(
        inputs={"text": torch.randn(2, 2, 2), "audio": torch.randn(2, 2, 2)},
        targets=None,
        masks={"text_mask": torch.ones(2), "audio_mask": torch.ones(2)},
    )

    restored = _identity_tfs()
    restored.load_state_dict(model.state_dict())

    assert torch.equal(model(batch).preds, restored(batch).preds)


def test_tfs_default_configuration_builds_pairwise_hierarchy():
    model = _identity_tfs()

    assert model.groups == (
        (("text", "audio"), ("audio", "text")),
        (("text", "video"), ("video", "text")),
        (("audio", "video"), ("video", "audio")),
    )


def test_tfs_default_backbone_has_masai_hierarchical_structure():
    model = _identity_tfs()

    assert model.triple_fusion is not None
    assert len(model.triple_fusion.blocks) == 6
    assert len(model.triple_fusion.cross_blocks) == 2
    assert model.triple_fusion.cross_att_dense_layer.in_features == 24
    assert all(hasattr(block, "positional_encoding") for block in model.triple_fusion.blocks)


def test_tfs_passes_unscaled_triples_through_both_second_level_blocks():
    class EchoBlock(nn.Module):
        def forward(self, query, key, query_valid, key_valid):
            return query * query_valid.unsqueeze(-1).to(query.dtype)

    model = TFSFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=1,
        head=nn.Identity(),
    ).eval()
    backbone = model.triple_fusion
    assert backbone is not None
    backbone.blocks = nn.ModuleList(EchoBlock() for _ in backbone.blocks)
    backbone.cross_blocks = nn.ModuleList(EchoBlock() for _ in backbone.cross_blocks)
    captured = []
    hooks = [
        block.register_forward_hook(lambda _, inputs, __: captured.append((inputs[0].clone(), inputs[1].clone())))
        for block in backbone.cross_blocks
    ]
    try:
        features = {
            "text": (torch.tensor([[[1.0]]]), torch.ones(1, 1, dtype=torch.bool)),
            "audio": (torch.tensor([[[2.0]]]), torch.ones(1, 1, dtype=torch.bool)),
            "video": (torch.tensor([[[3.0]]]), torch.ones(1, 1, dtype=torch.bool)),
        }
        _, _, _, _, pair_weights = backbone(features)
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.equal(pair_weights, torch.ones(1, 6))
    assert torch.equal(captured[0][0], torch.tensor([[[1.0, 2.0, 3.0]]]))
    assert torch.equal(captured[0][1], torch.tensor([[[2.0, 3.0, 1.0]]]))
    assert torch.equal(captured[1][0], torch.tensor([[[2.0, 3.0, 1.0]]]))
    assert torch.equal(captured[1][1], torch.tensor([[[1.0, 2.0, 3.0]]]))


def test_tfs_second_level_blocks_receive_finite_gradients():
    model = TFSFusionModel(
        encoders={"text": nn.Identity(), "audio": nn.Identity(), "video": nn.Identity()},
        shared_dim=2,
        head=nn.Identity(),
    )
    batch = Batch(
        inputs={
            "text": torch.randn(2, 1, 2),
            "audio": torch.randn(2, 1, 2),
            "video": torch.randn(2, 1, 2),
        },
        targets=None,
    )

    model(batch).aux["fused"].sum().backward()

    for block in model.triple_fusion.cross_blocks:
        gradients = [parameter.grad for parameter in block.parameters()]
        assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
