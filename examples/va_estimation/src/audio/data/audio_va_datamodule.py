from pathlib import Path

from audio.data.audio_va_dataset import AudioVADataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule
from chimera_ml.data.masking_collate import masking_collate


@DATAMODULES.register("audio_va_datamodule")
def audio_va_datamodule(
    *,
    dataset_path: str,
    train_csv: str,
    val_csv: str | None = None,
    test_csv: str | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    filter_non_speech: bool = True,
    augment: bool = True,
    augment_params: dict[str, float] | None = None,
    s2s: bool = False,
    s2s_steps: int = 4,
    **_,
) -> DataModule:
    """Audio VA DataModule.

    Provides:
      - train loader from `train_csv` (optionally filtered by `use_for_audio`)
      - validation loaders (if `val_csv` is provided):
          * "val"      : filtered (use_for_audio=True) — original audio-focused validation
          * "val_full" : unfiltered (all windows)       — for frame-wise / alternative validation
      - optional test loader (if `test_csv` is provided), unfiltered by default

    Note: chimera_ml's base DataModule supports multi-loader validation/test
    when val_dataset/test_dataset are passed as a dict[str, Dataset].
    """

    dataset_path = Path(dataset_path)
    augment_params = dict(augment_params or {})

    train_ds = None
    train_ds = AudioVADataset(
        csv_path=train_csv, 
        wav_root=dataset_path / "train",
        filter_non_speech=bool(filter_non_speech),
        labeled=True,
        split="train",
        augment=augment,
        augment_params=augment_params,
        s2s=s2s,
        s2s_steps=s2s_steps
    )

    val_ds = None
    if val_csv:
        val_ds = {
            # filtered validation (audio-only windows)
            "val": AudioVADataset(
                csv_path=val_csv,
                wav_root=dataset_path / "val",
                filter_non_speech=True,
                labeled=True,
                split="val",
                augment=False,
                s2s=s2s,
                s2s_steps=s2s_steps
            ),
            # unfiltered validation (all windows; used by frame-wise callback)
            "val_full": AudioVADataset(
                csv_path=val_csv,
                wav_root=dataset_path / "val",
                filter_non_speech=False,
                labeled=True,
                split="val_full",
                augment=False,
                s2s=s2s,
                s2s_steps=s2s_steps
            ),
        }

    test_ds = None
    if test_csv:
        test_ds = AudioVADataset(
            csv_path=Path(test_csv),
            wav_root=dataset_path / "test",
            filter_non_speech=False,
            labeled=False,
            split="test",
            augment=False,
            s2s=s2s,
            s2s_steps=s2s_steps
        )

    return DataModule(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=masking_collate(),
    )
