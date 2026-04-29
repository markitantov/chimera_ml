from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from common.utils import DatasetType, FeaturesType, compute_class_weights, load_pickle
from fusion.augmentation.modality_augmentation import ModalityDropAugmentation
from fusion.data.agender_multimodal_dataset import AGenderMultimodalDataset
from torch.utils.data import ConcatDataset, Dataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule


@dataclass
class AgenderFusionDataModule(DataModule):
    corpora: dict[str, dict[str, Any]] | None = None
    features_root: str = ""
    gender_class_names: list[str] | None = None
    include_mask: bool = False
    mask_class_names: list[str] | None = None
    channels: list[str] | None = None
    features_file_name: str = "SAMPLES"
    features_type: FeaturesType | int | str = FeaturesType.LATE
    dataset_type: DatasetType | int | str = DatasetType.BOTH
    augmentation: bool = False
    augmentation_params: dict[str, Any] | None = None
    audio_feature_extractor: dict[str, Any] | None = None
    image_feature_extractor: dict[str, Any] | None = None
    sr: int = 16000
    win_max_length: int = 4
    win_shift: int = 2
    win_min_length: int = 2
    train_subsets: tuple[str, ...] = ("train", "devel", "dev")
    val_subsets: tuple[str, ...] = ("test",)
    test_subsets: tuple[str, ...] | None = None

    def describe_context(self, context) -> None:
        context.set("data.gender_class_names", list(self.gender_class_names))
        context.set("data.gender_class_weights", list(self.gender_class_weights))

        context.set("data.include_mask", bool(self.include_mask))
        context.set("data.mask_class_names", list(self.mask_class_names or []))
        context.set("data.mask_class_weights", list(getattr(self, "mask_class_weights", [])))

        context.set("data.features_type", self.features_type)

    def __post_init__(self) -> None:
        if not self.corpora:
            raise ValueError("AgenderFusionDataModule requires a non-empty 'corpora' mapping.")

        if self.gender_class_names is not None:
            self.gender_class_names = [str(name) for name in self.gender_class_names]
            self.gender_num_classes = len(self.gender_class_names)

        if self.include_mask and self.mask_class_names is None:
            raise ValueError("AgenderFusionDataModule requires a mask class names")

        if self.include_mask and self.mask_class_names is not None:
            self.mask_class_names = [str(name) for name in self.mask_class_names]
            self.mask_num_classes = len(self.mask_class_names)

        train_datasets = []
        val_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}
        self.datasets_stats: dict[str, dict[str, Any]] = {}
        train_gen_counts: list[np.ndarray] = []

        if self.include_mask:
            train_mask_counts: list[np.ndarray] = []

        for corpus_name, cfg in self.corpora.items():
            labels = pd.read_csv(cfg["labels_file_path"])
            labels["subset"] = labels["subset"].astype(str)
            if "voxceleb" in corpus_name.lower():
                labels["subset"] = labels["subset"].replace({"dev": "train", "test": "dev"})

            vad = load_pickle(cfg.get("vad_path")) if cfg.get("vad_path") else None
            common = {
                "data_root": cfg["data_root"],
                "features_root": self.features_root,
                "corpus_name": corpus_name,
                "gender_num_classes": self.gender_num_classes,
                "include_mask": self.include_mask,
                "channels": self.channels,
                "vad_metadata": vad,
                "sr": self.sr,
                "win_max_length": self.win_max_length,
                "win_shift": self.win_shift,
                "win_min_length": self.win_min_length,
                "dataset_type": self.dataset_type,
                "features_type": self.features_type,
                "audio_feature_extractor_cfg": self.audio_feature_extractor,
                "image_feature_extractor_cfg": self.image_feature_extractor,
            }
            self.datasets_stats.setdefault(corpus_name, {})

            for subset_name in self.train_subsets:
                subset_name = str(subset_name)
                subset_labels = labels[labels["subset"] == subset_name].copy()
                if subset_labels.empty:
                    continue

                train_dataset = AGenderMultimodalDataset(
                    labels_metadata=subset_labels,
                    features_file_name=f"{corpus_name}_{subset_name.upper()}_{self.features_file_name}",
                    transform=(
                        ModalityDropAugmentation(**(self.augmentation_params or {})) if self.augmentation else None
                    ),
                    **common,
                )
                train_datasets.append(train_dataset)
                self.datasets_stats[corpus_name][subset_name] = train_dataset.stats
                train_gen_counts.append(train_dataset.stats["counts"]["gen"])
                if self.include_mask:
                    train_mask_counts.append(train_dataset.stats["counts"]["mask"])

            for subset_name in self.val_subsets:
                subset_name = str(subset_name)
                subset_labels = labels[labels["subset"] == subset_name].copy()
                if subset_labels.empty:
                    continue

                dataset_key = f"{subset_name}/{corpus_name}"
                val_datasets[dataset_key] = AGenderMultimodalDataset(
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
                    test_datasets[dataset_key] = AGenderMultimodalDataset(
                        labels_metadata=subset_labels,
                        features_file_name=f"{corpus_name}_{subset_name.upper()}_{self.features_file_name}",
                        **common,
                    )
                    self.datasets_stats[corpus_name][subset_name] = test_datasets[dataset_key].stats

        self.train_dataset = ConcatDataset(train_datasets) if train_datasets else None
        self.val_dataset = val_datasets or None
        self.test_dataset = test_datasets or None

        if train_gen_counts:
            counts = np.sum(train_gen_counts, axis=0)
            self.gender_class_weights = compute_class_weights(counts)

        else:
            self.gender_class_weights = [1.0] * int(self.gender_num_classes)

        if self.include_mask:
            if train_mask_counts:
                counts = np.sum(train_mask_counts, axis=0)
                self.mask_class_weights = compute_class_weights(counts)
            else:
                self.mask_class_weights = [1.0] * int(self.mask_num_classes)


@DATAMODULES.register("agender_fusion_datamodule")
def agender_fusion_datamodule(**params):
    return AgenderFusionDataModule(**params)
