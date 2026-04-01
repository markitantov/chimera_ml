from pathlib import Path
from typing import Any

from fusion.data.fusion_masking_collate import fusion_masking_collate
from fusion.data.fusion_va_dataset import FusionVADataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule


def split_mods(modalities: dict[str, dict[str, Any]] | None, split: str) -> tuple[dict[str, int], dict[str, Path]]:
    paths: dict[str, Path] = {}
    emb_sizes: dict[str, int] = {}
    for name, p in (modalities or {}).items():
        paths[name] = Path(p.get("path")) / f"{split}.pkl"
        emb_sizes[name] = int(p.get("emb_size"))

    return emb_sizes, paths


@DATAMODULES.register("fusion_va_datamodule")
def fusion_va_datamodule(
    *,
    # audio window pickles
    audio: dict[str, Any],
    use_predictions: bool,
    modalities: dict[str, dict[str, Any]] | None = None,
    frame_index_offset: int = 1,
    # audio masking
    mask_audio: bool = True,
    audio_min_open_sec: float = 1.0,
    audio_min_coverage_ratio: float = 0.8,
    # dataloader
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    **_,
) -> DataModule:
    tr_emb_sizes, tr_paths = split_mods(modalities, "train")
    va_emb_sizes, va_paths = split_mods(modalities, "val")
    te_emb_sizes, te_paths = split_mods(modalities, "test")

    train_ds = FusionVADataset(
        audio_windows_pkl=Path(audio["path"]) / "train.pkl",
        audio_emb_size=audio["emb_size"],
        use_predictions=bool(use_predictions),
        frame_modalities=tr_paths,
        modalities_emb_sizes=tr_emb_sizes,
        frame_index_offset=int(frame_index_offset),
        labeled=True,
        mask_audio=bool(mask_audio),
        audio_min_open_sec=float(audio_min_open_sec),
        audio_min_coverage_ratio=float(audio_min_coverage_ratio),
    )

    val_ds = {
        "val": FusionVADataset(
            audio_windows_pkl=Path(audio["path"]) / "val.pkl",
            audio_emb_size=audio["emb_size"],
            use_predictions=bool(use_predictions),
            frame_modalities=va_paths,
            modalities_emb_sizes=va_emb_sizes,
            frame_index_offset=int(frame_index_offset),
            labeled=True,
        )
    }

    test_ds = FusionVADataset(
        audio_windows_pkl=Path(audio["path"]) / "test.pkl",
        audio_emb_size=audio["emb_size"],
        use_predictions=bool(use_predictions),
        frame_modalities=te_paths,
        modalities_emb_sizes=te_emb_sizes,
        frame_index_offset=int(frame_index_offset),
        labeled=False,
    )

    return DataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=fusion_masking_collate(),
    )
