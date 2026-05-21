from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from common.utils import (
    compute_class_weights,
    load_pickle,
)
from feature_extractors.audio_features import AudioFeatureExtractor
from feature_extractors.text_features import TextFeatureExtractor
from feature_extractors.video_features import VideoFeatureExtractor
from fusion.augmentation.modality_augmentation import ModalityDropAugmentation
from fusion.data.emotion_multimodal_dataset import EmotionMultimodalDataset
from torch.utils.data import ConcatDataset, Dataset

from chimera_ml.core.registry import DATAMODULES
from chimera_ml.data.datamodule import DataModule


@dataclass(kw_only=True)
class EmotionMultimodalDataModule(DataModule):
    features_root: str
    feature_types: dict[str, str]
    features_file_name: str = "SAMPLES"
    corpora: dict[str, dict[str, Any]] | None = None

    emotion_class_names: list[str] | None = None
    sentiment_class_names: list[str] | None = None

    load_in_ram: bool = False
    augmentation: bool = False
    augmentation_params: dict[str, Any] | None = None

    sr: int = 16000
    win_max_length: int = 3
    win_shift: int = 2
    win_min_length: int = 2
    text_max_length: int = 48
    video_target_fps: int = 10

    train_subsets: tuple[str, ...] = ("train",)
    val_subsets: tuple[str, ...] = ("devel", "test")
    test_subsets: tuple[str, ...] | None = None

    def describe_context(self, context: Any) -> None:
        context.set("data.a_size", tuple(self.feature_sizes["a"]))
        context.set("data.v_size", tuple(self.feature_sizes["v"]))
        context.set("data.t_size", tuple(self.feature_sizes["t"]))

        context.set("data.emotion_class_names", list(self.emotion_class_names))
        context.set("data.sentiment_class_names", list(self.sentiment_class_names))

        context.set("data.emotion_class_weights", list(self.emotion_class_weights))
        context.set("data.sentiment_class_weights", list(self.sentiment_class_weights))

    def __post_init__(self) -> None:
        self.emotion_class_names = [str(name) for name in self.emotion_class_names]
        self.sentiment_class_names = [str(name) for name in self.sentiment_class_names]

        train_datasets: list[Dataset] = []
        val_datasets: dict[str, Dataset] = {}
        test_datasets: dict[str, Dataset] = {}
        self.datasets_stats: dict[str, dict[str, Any]] = {}
        train_emo_counts: list[np.ndarray] = []
        train_sen_counts: list[np.ndarray] = []

        for corpus_name, cfg in self.corpora.items():
            feature_extractors = self._init_feature_extractors(
                self.feature_types,
                self.sr,
                self.win_max_length,
                self.text_max_length,
                self.video_target_fps,
            )

            common = {
                "corpus_name": corpus_name,
                "features_root": self.features_root,
                "feature_types": self.feature_types,
                "emotion_num_classes": len(self.emotion_class_names),
                "sentiment_num_classes": len(self.sentiment_class_names),
                "sr": self.sr,
                "win_max_length": self.win_max_length,
                "win_shift": self.win_shift,
                "win_min_length": self.win_min_length,
                "load_in_ram": self.load_in_ram,
                "feature_extractors": feature_extractors,
            }

            self.datasets_stats.setdefault(corpus_name, {})

            for subset_name in self.train_subsets:
                subset_name = str(subset_name)
                subset_labels = pd.read_csv(str(cfg["labels_file_path"]).format(subset=subset_name or ""))

                train_dataset = self._build_dataset(
                    cfg=cfg,
                    subset_name=subset_name,
                    subset_labels=subset_labels,
                    corpus_name=corpus_name,
                    features_file_name=self.features_file_name,
                    transform=(
                        ModalityDropAugmentation(**(self.augmentation_params or {})) if self.augmentation else None
                    ),
                    common=common,
                )

                train_datasets.append(train_dataset)
                self.datasets_stats[corpus_name][subset_name] = train_dataset.stats
                train_emo_counts.append(train_dataset.stats["counts"]["emo"])
                train_sen_counts.append(train_dataset.stats["counts"]["sen"])

            for subset_name in self.val_subsets:
                subset_name = str(subset_name)
                subset_labels = pd.read_csv(str(cfg["labels_file_path"]).format(subset=subset_name or ""))

                dataset_key = f"{subset_name}/{corpus_name}"

                val_datasets[dataset_key] = self._build_dataset(
                    cfg=cfg,
                    subset_name=subset_name,
                    subset_labels=subset_labels,
                    corpus_name=corpus_name,
                    features_file_name=self.features_file_name,
                    transform=None,
                    common=common,
                )

                self.datasets_stats[corpus_name][str(subset_name)] = val_datasets[dataset_key].stats

            if self.test_subsets:
                for subset_name in self.test_subsets:
                    subset_name = str(subset_name)
                    subset_labels = pd.read_csv(str(cfg["labels_file_path"]).format(subset=subset_name or ""))

                    dataset_key = f"{subset_name}/{corpus_name}"

                    test_datasets[dataset_key] = self._build_dataset(
                        cfg=cfg,
                        subset_name=subset_name,
                        subset_labels=subset_labels,
                        corpus_name=corpus_name,
                        features_file_name=self.features_file_name,
                        transform=None,
                        common=common,
                    )

                    self.datasets_stats[corpus_name][subset_name] = test_datasets[dataset_key].stats

        self.train_dataset = ConcatDataset(train_datasets) if train_datasets else None
        self.val_dataset = val_datasets or None
        self.test_dataset = test_datasets or None

        if train_emo_counts:
            self.emotion_class_weights = compute_class_weights(np.sum(train_emo_counts, axis=0))
        else:
            self.emotion_class_weights = [1.0] * len(self.emotion_class_names)

        if train_sen_counts:
            self.sentiment_class_weights = compute_class_weights(np.sum(train_sen_counts, axis=0))
        else:
            self.sentiment_class_weights = [1.0] * len(self.sentiment_class_names)

    def _build_dataset(
        self,
        cfg: dict[str, Any],
        subset_name: str,
        subset_labels: pd.DataFrame,
        corpus_name: str,
        features_file_name: str,
        transform: Any | None,
        common: dict[str, Any],
    ) -> EmotionMultimodalDataset | None:
        audio_root = str(cfg["audio_root"]).format(subset=subset_name or "")
        video_root = str(cfg["video_root"]).format(subset=subset_name or "")
        vad_path = str(cfg["vad_path"]).format(subset=subset_name or "")
        vad_metadata = load_pickle(vad_path) if vad_path else None

        return EmotionMultimodalDataset(
            labels_metadata=subset_labels,
            audio_root=audio_root,
            video_root=video_root,
            vad_metadata=vad_metadata,
            subset_name=subset_name,
            features_file_name=f"{corpus_name}_{subset_name.upper()}_{features_file_name}",
            transform=transform,
            **common,
        )

    def _init_feature_extractors(
        self,
        feature_types: dict[str, str],
        sr: int = 16000,
        win_max_length: int = 3,
        text_max_length: int = 48,
        video_target_fps: int = 10,
    ) -> dict[str, Any]:
        res = {}
        self.feature_sizes = {}
        for modality, feature_type in feature_types.items():
            if modality.lower() == "a":
                fe = AudioFeatureExtractor(sr=sr, win_max_length=win_max_length, model_name=feature_type)
            elif modality.lower() == "v":
                fe = VideoFeatureExtractor(
                    win_max_length=win_max_length, target_fps=video_target_fps, model_name=feature_type
                )
            elif modality.lower() == "t":
                fe = TextFeatureExtractor(max_length=text_max_length, model_name=feature_type)
            else:
                raise NotImplementedError

            res[modality] = fe
            self.feature_sizes[modality] = fe.output_shape

        return res


@DATAMODULES.register("emotion_multimodal_datamodule")
def emotion_multimodal_datamodule(**params: Any) -> EmotionMultimodalDataModule:
    return EmotionMultimodalDataModule(**params)
