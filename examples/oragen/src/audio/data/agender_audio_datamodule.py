from dataclasses import dataclass
from typing import Any

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule


from common.utils import load_pickle
from audio.data.agender_audio_dataset import AGenderAudioDataset

@dataclass
class AgenderAudioDataModule(DataModule):
    corpora: dict[str, dict[str, Any]] | None = None
    features_root: str
    gender_num_classes: int
    features_file_name: str = "SAMPLES"
    sr: int = 16000
    win_max_length: int = 4
    win_shift: int = 2
    win_min_length: int = 2
    preprocessor_name: str | None = None
    train_subsets: tuple[str, ...] = ("train", "devel", "dev")
    val_subsets: tuple[str, ...] = ("test",)
    test_subsets: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("AgenderAudioDataModule requires a non-empty 'corpora' mapping.")

        train_datasets = []
        val_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}

        for corpus_name, cfg in self.corpora.items():
            labels = pd.read_csv(cfg["labels_csv"])
            vad = load_pickle(cfg.get("vad_path")) if cfg.get("vad_path") else None
            common = {
                "data_root": cfg["data_root"],
                "features_root": self.features_root,
                "corpus_name": corpus_name,
                "gender_num_classes": int(cfg.get("gender_num_classes", self.gender_num_classes)),
                "vad_metadata": vad,
                "sr": self.sr,
                "win_max_length": self.win_max_length,
                "win_shift": self.win_shift,
                "win_min_length": self.win_min_length,
                "preprocessor_name": self.preprocessor_name,
            }

            train_datasets.append(
                AGenderAudioDataset(
                    labels_metadata=labels[labels["subset"].isin(list(self.train_subsets))].copy(),
                    features_file_name=f"{corpus_name}_TRAIN_{self.features_file_name}",
                    **common,
                )
            )

            val_datasets[corpus_name] = AGenderAudioDataset(
                labels_metadata=labels[labels["subset"].isin(list(self.val_subsets))].copy(),
                features_file_name=f"{corpus_name}_VAL_{self.features_file_name}",
                **common,
            )

            if self.test_subsets:
                test_datasets[corpus_name] = AGenderAudioDataset(
                    labels_metadata=labels[labels["subset"].isin(list(self.test_subsets))].copy(),
                    features_file_name=f"{corpus_name}_TEST_{self.features_file_name}",
                    **common,
                )

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = val_datasets
        self.test_dataset = test_datasets or None


@DATAMODULES.register("agender_audio_datamodule")
def agender_audio_datamodule(**params):
    return AgenderAudioDataModule(**params)
