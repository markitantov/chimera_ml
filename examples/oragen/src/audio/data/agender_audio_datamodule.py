from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from audio.data.agender_audio_dataset import AGenderAudioDataset
from common.utils import load_pickle
from torch.utils.data import ConcatDataset, Dataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule


@dataclass
class AgenderAudioDataModule(DataModule):
    corpora: dict[str, dict[str, Any]] | None = None
    features_root: str = ""
    gender_class_names: list[str] | None = None
    features_file_name: str = "SAMPLES"
    sr: int = 16000
    win_max_length: int = 4
    win_shift: int = 2
    win_min_length: int = 2
    preprocessor_name: str | None = None
    train_subsets: tuple[str, ...] = ("train", "devel", "dev")
    val_subsets: tuple[str, ...] = ("test",)
    test_subsets: tuple[str, ...] | None = None

    def describe_context(self, context) -> None:
        context.set("data.gender_class_names", list(self.gender_class_names))
        context.set("data.gender_class_weights", list(self.gender_class_weights))
        context.set("data.win_max_length", int(self.win_max_length))

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("AgenderAudioDataModule requires a non-empty 'corpora' mapping.")
        
        if self.gender_class_names is not None:
            self.gender_class_names = [str(name) for name in self.gender_class_names]
            self.gender_num_classes = len(self.gender_class_names)

        train_datasets = []
        val_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}
        self.datasets_stats: dict[str, dict[str, Any]] = {}
        train_counts: list[np.ndarray] = []

        for corpus_name, cfg in self.corpora.items():
            labels = pd.read_csv(cfg["labels_file_path"])
            labels["subset"] = labels["subset"].astype(str)
            vad = load_pickle(cfg.get("vad_path")) if cfg.get("vad_path") else None
            common = {
                "data_root": cfg["data_root"],
                "features_root": self.features_root,
                "corpus_name": corpus_name,
                "gender_num_classes": self.gender_num_classes,
                "vad_metadata": vad,
                "sr": self.sr,
                "win_max_length": self.win_max_length,
                "win_shift": self.win_shift,
                "win_min_length": self.win_min_length,
                "preprocessor_name": self.preprocessor_name,
            }
            self.datasets_stats.setdefault(corpus_name, {})

            for subset_name in self.train_subsets:
                subset_name = str(subset_name)
                subset_labels = labels[labels["subset"] == subset_name].copy()
                if subset_labels.empty:
                    continue

                train_dataset = AGenderAudioDataset(
                    labels_metadata=subset_labels,
                    features_file_name=f"{corpus_name}_{subset_name.upper()}_{self.features_file_name}",
                    **common,
                )
                train_datasets.append(train_dataset)
                self.datasets_stats[corpus_name][subset_name] = train_dataset.stats
                train_counts.append(train_dataset.stats["counts"]["gen"])

            for subset_name in self.val_subsets:
                subset_name = str(subset_name)
                subset_labels = labels[labels["subset"] == subset_name].copy()
                if subset_labels.empty:
                    continue

                dataset_key = f"{subset_name}/{corpus_name}"
                val_datasets[dataset_key] = AGenderAudioDataset(
                    labels_metadata=subset_labels,
                    features_file_name=f"{corpus_name}_{subset_name.upper()}_{self.features_file_name}",
                    **common,
                )
                self.datasets_stats[corpus_name][subset_name] = val_datasets[dataset_key].stats

            if self.test_subsets:
                for subset_name in self.test_subsets:
                    subset_name = str(subset_name)
                    subset_labels = labels[labels["subset"] == subset_name].copy()
                    if subset_labels.empty:
                        continue
                    
                    dataset_key = f"{subset_name}/{corpus_name}"
                    test_datasets[dataset_key] = AGenderAudioDataset(
                        labels_metadata=subset_labels,
                        features_file_name=f"{corpus_name}_{subset_name.upper()}_{self.features_file_name}",
                        **common,
                    )
                    self.datasets_stats[corpus_name][subset_name] = test_datasets[dataset_key].stats

        self.train_dataset = ConcatDataset(train_datasets) if train_datasets else None
        self.val_dataset = val_datasets or None
        self.test_dataset = test_datasets or None

        if train_counts:
            counts = np.sum(train_counts, axis=0)
            total = counts.sum()
            self.gender_class_weights = (
                (counts / total).tolist()
                if total > 0
                else [1.0] * len(counts)
            )
        else:
            self.gender_class_weights = [1.0] * self.gender_num_classes

@DATAMODULES.register("agender_audio_datamodule")
def agender_audio_datamodule(context = None, **params):
    return AgenderAudioDataModule(**params)
