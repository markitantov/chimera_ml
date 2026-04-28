import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from common.utils import (
    DatasetType,
    FeaturesType,
    find_intersections,
    gender_label_to_int,
    generate_features_suffix,
    load_pickle,
    mask_label_to_int,
    normalize_audio_filename,
    read_audio,
    save_pickle,
    slice_audio,
    waveform_cache_name,
)
from fusion.data.feature_extractors import AudioFeatureExtractor, ImageFeatureExtractor
from torch.utils.data import Dataset
from tqdm import tqdm


def _enum_value(enum_cls, value):
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        key = value.strip().upper()
        if key in enum_cls.__members__:
            return enum_cls[key]

    return enum_cls(int(value))


class AGenderMultimodalDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str,
        labels_metadata: pd.DataFrame,
        features_root: str,
        features_file_name: str,
        corpus_name: str,
        gender_num_classes: int,
        include_mask: bool = False,
        mask_num_classes: int = 0,
        channels: list[str] | None = None,
        vad_metadata: dict[str, list[dict[str, int]]] | None = None,
        sr: int = 16000,
        win_max_length: int = 4,
        win_shift: int = 2,
        win_min_length: int = 0,
        features_type: FeaturesType | int | str = FeaturesType.LATE,
        dataset_type: DatasetType | int | str = DatasetType.BOTH,
        audio_feature_extractor_cfg: dict[str, Any] | None = None,
        image_feature_extractor_cfg: dict[str, Any] | None = None,
        transform: Any = None,
        display_filtering_stats: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.labels_metadata = labels_metadata
        self.vad_metadata = vad_metadata
        self.sr = int(sr)
        self.win_max_length = int(win_max_length)
        self.win_shift = int(win_shift)
        self.win_min_length = int(win_min_length)
        self.transform = transform
        self.corpus_name = corpus_name
        self.gender_num_classes = int(gender_num_classes)
        self.include_mask = bool(include_mask)
        self.mask_num_classes = int(mask_num_classes)
        self.channels = channels or ["c"]

        self.dataset_type = _enum_value(DatasetType, dataset_type)
        self.features_type = _enum_value(FeaturesType, features_type)

        self.audio_feature_extractor_cfg = audio_feature_extractor_cfg
        self.image_feature_extractor_cfg = image_feature_extractor_cfg

        self.display_filtering_stats = display_filtering_stats

        self.audio_feature_extractor: AudioFeatureExtractor | None = None
        self.image_feature_extractor: ImageFeatureExtractor | None = None

        suffix = generate_features_suffix(
            vad_metadata=self.vad_metadata,
            win_max_length=self.win_max_length,
            win_shift=self.win_shift,
            win_min_length=self.win_min_length,
        )

        typed_features_file_name = f"{features_file_name}_{self.features_type.name}"
        self.full_features_path = Path(features_root) / f"{typed_features_file_name}_{suffix}"
        self.full_features_path.mkdir(parents=True, exist_ok=True)
        stats_path = Path(features_root) / f"{typed_features_file_name}_{suffix}_stats.pickle"

        info = load_pickle(stats_path)
        if not info:
            print("No cached multimodal feature index found. Start preparing fusion features.")
            info = self._prepare_data()
            save_pickle(info, stats_path)
            print(f"Feature index saved to '{stats_path}'.")
        else:
            print(f"Using cached multimodal feature index from '{stats_path}'.")

        self.info, self.stats = self._filter_samples(info)
        print(
            f"Ready: num_windows={len(self.info)}, "
            f"num_files={len(self.stats['fns'])}, "
            f"gender_counts={self.stats['counts']['gen'].tolist()}"
        )

    def init_feature_extractors(self) -> None:
        if self.audio_feature_extractor_cfg is None:
            raise ValueError("audio_feature_extractor config is required to compute missing multimodal features.")

        self.audio_feature_extractor = AudioFeatureExtractor(
            hf_model_name=self.audio_feature_extractor_cfg["hf_model_name"],
            checkpoint_path=self.audio_feature_extractor_cfg["checkpoint_path"],
            features_type=self.features_type,
            sr=self.sr,
            win_max_length=self.win_max_length,
            gender_num_classes=self.gender_num_classes,
        )

        if self.image_feature_extractor_cfg is None:
            raise ValueError("image_feature_extractor config is required to compute missing multimodal features.")

        self.image_feature_extractor = ImageFeatureExtractor(
            hf_model_name=self.audio_feature_extractor_cfg["hf_model_name"],
            checkpoint_path=self.audio_feature_extractor_cfg["checkpoint_path"],
            features_type=self.features_type,
        )

    def _prepare_data(self) -> list[dict[str, Any]]:
        info: list[dict[str, Any]] = []

        self.init_feature_extractors()

        records = (
            self.labels_metadata.sort_values("audio_file_path").drop_duplicates(["audio_file_path"], keep="last").copy()
        ).to_dict("records")
        files_without_windows_before_vad = 0
        files_without_windows_after_vad = 0
        windows_before_vad = 0
        windows_after_vad = 0

        for sample in tqdm(records, desc=f"{self.corpus_name}: extracting multimodal features"):
            sample_filename = normalize_audio_filename(sample["audio_file_path"])
            sample_fp = self.data_root / sample_filename
            full_wave = read_audio(sample_fp, self.sr)
            windows = slice_audio(
                start_time=0,
                end_time=len(full_wave),
                win_max_length=int(self.win_max_length * self.sr),
                win_shift=int(self.win_shift * self.sr),
                win_min_length=int(self.win_min_length * self.sr),
            )

            if not windows:
                files_without_windows_before_vad += 1
                continue

            windows_before_vad += len(windows)

            if self.vad_metadata:
                windows = find_intersections(
                    windows,
                    self.vad_metadata[sample_filename],
                    min_length=int(self.win_min_length * self.sr),
                )
                if not windows:
                    files_without_windows_after_vad += 1
                    continue

            windows_after_vad += len(windows)

            if (
                ("lQxVumsa0QE/00053" in sample_filename)
                or ("x9K8-IfuOMg/00490" in sample_filename)
                or ("x9K8-IfuOMg/00491" in sample_filename)
                or ("StiTPpXXhe0/00079" in sample_filename)
            ):
                continue

            for w_idx, window in enumerate(windows):
                wave = full_wave[window["start"] : window["end"]].clone()
                images_fn, images_fp = self._window_images(sample, window)
                info.append(
                    {
                        "fp": str(sample_fp),
                        "fn": sample_filename,
                        "img_fp": [str(path) for path in images_fp],
                        "img_fn": images_fn,
                        "w_idx": w_idx,
                        "start": window["start"],
                        "end": window["end"],
                        "gen": sample["gender"],
                        "age": sample["age"],
                        "mask": sample.get("mask_type", "No mask"),
                        "channel": sample.get("channel", "c"),
                    }
                )

                features = {
                    "acoustic_features": self.audio_feature_extractor(wave),
                    "visual_features": self.image_feature_extractor([str(path) for path in images_fp]),
                }

                save_pickle(features, self.full_features_path / waveform_cache_name(sample_filename, w_idx))

        if self.display_filtering_stats:
            print(
                f"Prepared {len(info)} window records from {len(records)} source rows. "
                f"files_without_windows_before_vad={files_without_windows_before_vad}, "
                f"files_without_windows_after_vad={files_without_windows_after_vad}, "
                f"windows_before_vad={windows_before_vad}, "
                f"windows_after_vad={windows_after_vad}"
            )

        return info

    def _window_images(self, sample: dict[str, Any], window: dict[str, int]) -> tuple[list[str], list[Path]]:
        image_parts = re.split(r"__s\d{3}", str(sample["image_file_path"]), maxsplit=1)

        start_second = int(window["start"] / self.sr) + 1
        end_second = int(window["end"] / self.sr) + 1
        images_fn = [
            f"{image_parts[0]}__s{frame_idx:03d}{image_parts[1]}" for frame_idx in range(start_second, end_second)
        ]

        images_fp = [self.data_root / image_fn for image_fn in images_fn]

        missing = [str(path) for path in images_fp if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing visual frames required for multimodal feature extraction: " + ", ".join(missing[:5])
            )

        return images_fn, images_fp

    def _filter_samples(self, info: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        stats = {
            "fns": {},
            "majority_class": {"gen": 0, "age": 0.0, "mask": 0},
            "counts": {
                "gen": np.zeros(self.gender_num_classes, dtype=np.int64),
                "age": 0.0,
                "mask": np.zeros(6, dtype=np.int64),
            },
        }

        new_info = []
        for sample in info:
            if "child" in str(sample["gen"]).lower() and self.gender_num_classes < 3:
                continue

            gen = gender_label_to_int(sample["gen"], self.gender_num_classes)
            age = float(sample["age"]) / 100.0
            mask = mask_label_to_int(sample["mask"])

            new_info.append({**sample, "gen": gen, "age": age, "mask": mask})
            stats["fns"][sample["fn"]] = {"gen": gen, "age": age, "mask": mask}
            stats["counts"]["gen"][gen] += 1
            stats["counts"]["age"] += age
            stats["counts"]["mask"][mask] += 1

        total = max(int(stats["counts"]["gen"].sum()), 1)
        stats["majority_class"]["gen"] = int(np.argmax(stats["counts"]["gen"]))
        stats["majority_class"]["age"] = float(stats["counts"]["age"] / total)
        stats["majority_class"]["mask"] = int(np.argmax(stats["counts"]["mask"]))
        return new_info, stats

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.info[index]
        features = load_pickle(self.full_features_path / waveform_cache_name(data["fn"], data["w_idx"]))
        if features is None:
            raise FileNotFoundError(self.full_features_path / waveform_cache_name(data["fn"], data["w_idx"]))

        audio = torch.as_tensor(features["acoustic_features"], dtype=torch.float32)
        image = torch.as_tensor(features["visual_features"], dtype=torch.float32)

        if self.dataset_type == DatasetType.VIDEO:
            audio = torch.zeros_like(audio)
        elif self.dataset_type == DatasetType.AUDIO:
            image = torch.zeros_like(image)
        elif self.transform is not None:
            audio, image = self.transform((audio, image))

        target_values = [float(data["gen"]), float(data["age"])]
        if self.include_mask:
            target_values.append(float(data["mask"]))

        return {
            "inputs": {"audio": audio, "image": image},
            "target": torch.tensor(target_values, dtype=torch.float32),
            "meta": {
                "filename": data["fn"],
                "start_t": data["start"] / self.sr,
                "end_t": data["end"] / self.sr,
                "start_f": data["start"],
                "end_f": data["end"],
                "corpus_name": self.corpus_name,
            },
        }
