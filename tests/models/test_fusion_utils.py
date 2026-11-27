import pytest
import torch

from chimera_ml.core.batch import Batch
from chimera_ml.data.masking_collate import MaskingCollate
from chimera_ml.models.utils import (
    build_projections,
    get_modality_mask,
    normalize_sequence,
    prepare_projected_features,
    validate_input_dimensions,
)


def test_masking_collate_masks_are_combined_for_temporal_features():
    batch = MaskingCollate()(
        [
            {"inputs": {"text": torch.ones(2, 4)}},
            {"inputs": {}},
        ]
    )

    result = get_modality_mask(batch, "text", batch.inputs["text"])
    expected = batch.masks["sequence_mask"] & batch.masks["text_mask"].bool()[:, None]

    assert torch.equal(result, expected)


def test_get_modality_mask_accepts_sample_and_temporal_masks():
    values = torch.ones(2, 3, 4)
    batch = Batch(
        inputs={"audio": values},
        targets=None,
        masks={"audio_mask": torch.tensor([1, 0])},
    )

    result = get_modality_mask(batch, "audio", values)

    assert torch.equal(result, torch.tensor([[True, True, True], [False, False, False]]))


def test_get_modality_mask_rejects_invalid_shapes():
    values = torch.ones(2, 3, 4)

    with pytest.raises(ValueError, match="shape \[B, T, D\]"):
        get_modality_mask(Batch(inputs={}, targets=None), "audio", torch.ones(2, 4))

    with pytest.raises(ValueError, match="shape \(B,\) or \(B, T\)"):
        get_modality_mask(
            Batch(inputs={}, targets=None, masks={"audio_mask": torch.ones(2, 2)}),
            "audio",
            values,
        )


def test_normalize_sequence_converts_vectors_and_rejects_other_ranks():
    assert normalize_sequence(torch.ones(2, 4)).shape == (2, 1, 4)

    with pytest.raises(ValueError, match="Expected \[B, D\] or \[B, T, D\]"):
        normalize_sequence(torch.ones(2, 3, 4, 5))


def test_prepare_projected_features_projects_present_modalities():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dims = {"audio": 4}
            self.proj = torch.nn.ModuleDict({"audio": torch.nn.Linear(4, 3)})

    result = prepare_projected_features(
        Model(),
        Batch(inputs={"audio": torch.ones(2, 4)}, targets=None),
        ("audio",),
    )

    assert result["audio"][0].shape == (2, 1, 3)
    assert result["audio"][1].shape == (2, 1)


def test_validate_input_dimensions_and_build_projections_reject_invalid_config():
    with pytest.raises(ValueError, match="non-empty"):
        validate_input_dimensions({})

    with pytest.raises(ValueError, match="positive"):
        validate_input_dimensions({"audio": 0})

    with pytest.raises(ValueError, match="Missing input dimensions"):
        build_projections(("audio",), {}, 4)

    projections = build_projections(("audio",), {"audio": 4}, 3)

    assert projections["audio"].out_features == 3
