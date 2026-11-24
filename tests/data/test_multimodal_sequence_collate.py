import pytest
import torch

from chimera_ml.core.config import ExperimentConfig
from chimera_ml.core.registry import COLLATES
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.data.multimodal_sequence_collate import MultimodalSequenceCollate
from chimera_ml.training.builders import build_collate


def test_vectors_keep_feature_dimension_and_zero_values_are_valid():
    collate = MultimodalSequenceCollate()
    batch = collate(
        [
            {"inputs": {"text": torch.tensor([0.0, 0.0, 0.0])}},
            {"inputs": {"text": torch.tensor([1.0, 2.0, 3.0])}},
        ]
    )

    assert batch.inputs["text"].shape == (2, 3)
    assert torch.equal(batch.inputs["text"][0], torch.zeros(3))
    assert torch.equal(batch.masks["text_mask"], torch.tensor([1.0, 1.0]))
    assert torch.equal(batch.masks["text_temporal_mask"], torch.tensor([[True], [True]]))


def test_modalities_are_padded_independently_and_missing_rows_are_zero():
    batch = MultimodalSequenceCollate(pad_value=9.0)(
        [
            {
                "inputs": {
                    "audio": torch.ones(2, 2),
                    "video": torch.ones(1, 3),
                }
            },
            {"inputs": {"audio": torch.full((3, 2), 2.0)}},
        ]
    )

    assert batch.inputs["audio"].shape == (2, 3, 2)
    assert batch.inputs["video"].shape == (2, 1, 3)
    assert torch.equal(batch.inputs["audio"][0, 2], torch.full((2,), 9.0))
    assert torch.equal(batch.inputs["video"][1], torch.zeros(1, 3))
    assert torch.equal(batch.masks["video_mask"], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        batch.masks["audio_temporal_mask"],
        torch.tensor([[True, True, False], [True, True, True]]),
    )
    assert torch.equal(batch.masks["video_temporal_mask"], torch.tensor([[True], [False]]))


def test_explicit_temporal_masks_are_combined_with_padding():
    batch = MultimodalSequenceCollate()(
        [
            {
                "inputs": {"audio": torch.arange(6, dtype=torch.float32).reshape(3, 2)},
                "meta": {"masks": {"audio_temporal_mask": [True, False, True]}},
            },
            {
                "inputs": {"audio": torch.ones(1, 2)},
                "meta": {"masks": {"audio_temporal_mask": [True]}},
            },
        ]
    )

    assert torch.equal(
        batch.masks["audio_temporal_mask"],
        torch.tensor([[True, False, True], [True, False, False]]),
    )


@pytest.mark.parametrize(
    "samples",
    [
        [{"inputs": {"audio": torch.ones(2, 3)}}, {"inputs": {"audio": torch.ones(2, 3, 1)}}],
        [{"inputs": {"audio": torch.ones(2, 3)}}, {"inputs": {"audio": torch.ones(2, 4)}}],
    ],
)
def test_incompatible_modality_shapes_are_rejected(samples):
    with pytest.raises(ValueError, match="incompatible"):
        MultimodalSequenceCollate()(samples)


def test_pad_sequences_false_rejects_variable_temporal_lengths():
    samples = [
        {"inputs": {"audio": torch.ones(2, 3)}},
        {"inputs": {"audio": torch.ones(3, 3)}},
    ]

    with pytest.raises(ValueError, match="pad_sequences=False"):
        MultimodalSequenceCollate(pad_sequences=False)(samples)


def test_pad_sequences_false_stacks_equal_sequences_without_padding():
    batch = MultimodalSequenceCollate(pad_sequences=False)(
        [
            {"inputs": {"audio": torch.ones(2, 3)}},
            {"inputs": {"audio": torch.full((2, 3), 2.0)}},
        ]
    )

    assert batch.inputs["audio"].shape == (2, 2, 3)
    assert torch.equal(batch.masks["audio_temporal_mask"], torch.ones(2, 2, dtype=torch.bool))


def test_zero_length_sequences_have_false_temporal_masks():
    batch = MultimodalSequenceCollate()(
        [
            {"inputs": {"audio": torch.empty(0, 2)}},
            {"inputs": {"audio": torch.ones(2, 2)}},
        ]
    )

    assert batch.inputs["audio"].shape == (2, 2, 2)
    assert torch.equal(batch.masks["audio_temporal_mask"], torch.tensor([[False, False], [True, True]]))


def test_zero_length_sequences_are_supported_when_all_are_empty():
    batch = MultimodalSequenceCollate()(
        [{"inputs": {"audio": torch.empty(0, 2)}}, {"inputs": {"audio": torch.empty(0, 2)}}]
    )

    assert batch.inputs["audio"].shape == (2, 0, 2)
    assert batch.masks["audio_temporal_mask"].shape == (2, 0)


def test_invalid_explicit_temporal_mask_shape_is_rejected():
    samples = [
        {
            "inputs": {"audio": torch.ones(2, 3)},
            "meta": {"masks": {"audio_temporal_mask": [True]}},
        }
    ]

    with pytest.raises(ValueError, match="Explicit mask for 'audio'"):
        MultimodalSequenceCollate()(samples)


@pytest.mark.parametrize(
    ("registry_name", "expected_type"),
    [
        ("masking_collate", MaskingCollate),
        ("multimodal_sequence_collate", MultimodalSequenceCollate),
    ],
)
def test_collates_build_from_yaml_registry_configuration(tmp_path, registry_name, expected_type):
    path = tmp_path / f"{registry_name}.yaml"
    ExperimentConfig({"collate": {"name": registry_name, "params": {}}}).to_yaml(path)

    config = ExperimentConfig.from_yaml(str(path))
    collate = build_collate(config.section("collate"))

    assert isinstance(collate, expected_type)


def test_legacy_metadata_masks_are_optional_and_registry_keys_are_independent():
    batch = MultimodalSequenceCollate(include_legacy_meta_masks=True)([{"inputs": {"text": torch.ones(3)}}])

    assert batch.meta is not None
    assert batch.meta["masks"] is batch.masks
    assert COLLATES.get("masking_collate") is not COLLATES.get("multimodal_sequence_collate")
    assert isinstance(build_collate({"name": "masking_collate", "params": {}}), MaskingCollate)
    assert isinstance(build_collate({"name": "multimodal_sequence_collate", "params": {}}), MultimodalSequenceCollate)
