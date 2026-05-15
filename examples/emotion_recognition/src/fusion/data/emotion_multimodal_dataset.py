from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from common.utils import (
    emotion_to_target,
    generate_features_suffix,
    load_pickle,
    normalize_audio_filename,
    read_audio,
    save_pickle,
    sentiment_to_onehot,
    slice_audio,
)
from torch.utils.data import Dataset
from tqdm import tqdm


class EmotionMultimodalDataset(Dataset):
    def __init__(
        self,
        *,
        labels_metadata: pd.DataFrame,
        audio_root: str,
        video_root: str,
        corpus_name: str,
        subset_name: str,
        features_root: str,
        feature_types: dict[str, str],
        features_file_name: str,
        feature_extractors: dict[str, Any],
        emotion_num_classes: int = 7,
        sentiment_num_classes: int = 3,
        vad_metadata: dict[str, list[dict[str, int]]] | None = None,
        sr: int = 16000,
        win_max_length: int = 3,
        win_shift: int = 2,
        win_min_length: int = 2,
        load_in_ram: bool = False,
        transform: Any = None,
    ) -> None:

        self.audio_root = Path(audio_root)
        self.video_root = Path(video_root)

        self.labels_metadata = labels_metadata
        self.vad_metadata = vad_metadata
        self.sr = int(sr)
        self.win_max_length = int(win_max_length)
        self.win_shift = int(win_shift)
        self.win_min_length = int(win_min_length)
        self.transform = transform
        self.corpus_name = corpus_name
        self.subset_name = str(subset_name)
        self.emotion_num_classes = emotion_num_classes
        self.sentiment_num_classes = sentiment_num_classes
        self.load_in_ram = load_in_ram

        self.feature_extractors = feature_extractors

        self.full_features_paths, self.full_stats_path = self._prepare_caching(
            features_root, features_file_name, feature_types
        )

        info: dict[str, list[dict[str, Any]]] = {}

        cache_ok = True
        for modality, full_stats_path in self.full_stats_path.items():
            m_info = load_pickle(full_stats_path)

            if not m_info:
                cache_ok = False
                break

            info[modality] = m_info

        if not cache_ok:
            print("No cached feature indexes found. Start preparing features.")
            info = self._prepare_data()
            for modality, full_stats_path in self.full_stats_path.items():
                save_pickle(info[modality], full_stats_path)

                print(f"{modality} feature index saved to '{full_stats_path}'.")
        else:
            print("Using cached feature indexes.")

        self.info, self.stats = self._filter_samples(info["a"])

        print(
            f"Ready: num_windows={len(self.info)}, "
            f"num_files={len(self.stats['fns'])}, "
            f"sen_counts={self.stats['counts']['sen'].tolist()}"
        )

    def _prepare_caching(
        self, features_root: str, features_file_name: str, feature_types: dict[str, str]
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        full_features_path: dict[str, Path] = {}
        full_stats_path: dict[str, Path] = {}
        for modality, feature_type in feature_types.items():
            suffix = generate_features_suffix(
                vad_metadata=self.vad_metadata,
                win_max_length=self.win_max_length,
                win_shift=self.win_shift,
                win_min_length=self.win_min_length,
            )

            full_features_path[modality] = (
                Path(features_root) / f"{features_file_name}_{feature_type}_{suffix}_{modality}"
            )
            full_features_path[modality].mkdir(parents=True, exist_ok=True)
            full_stats_path[modality] = (
                Path(features_root) / f"{features_file_name}_{feature_type}_{suffix}_stats_{modality}.pickle"
            )

        return full_features_path, full_stats_path

    def _prepare_data(self) -> dict[str, list[dict[str, Any]]]:
        info: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

        for sample in tqdm(
            self.labels_metadata.to_dict("records"),
            desc=f"{self.corpus_name}: preparing multimodal cache",
        ):
            sample_audio_path = self.audio_root / normalize_audio_filename(sample["video_name"])
            sample_filename = (
                f"{sample['video_name']}_{sample['start_time']}_{sample['end_time']}.wav"
                if "CMUMOSEI" in self.corpus_name
                else (sample_audio_path.name)
            )

            if "RAMAS" in self.corpus_name:
                sample_video_path = self.video_root / sample_filename.replace(".wav", ".mov")
            elif "CMUMOSEI" in self.corpus_name:
                sample_video_path = self.video_root / (
                    f"{sample['video_name']}_{float(sample['start_time']):.4f}_{float(sample['end_time']):.4f}.mp4"
                )
            else:
                sample_video_path = self.video_root / sample_filename.replace(".wav", ".mp4")

            sample_emo = emotion_to_target(
                [
                    float(sample[emotion_name])
                    for emotion_name in ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]
                ],
                self.emotion_num_classes,
            )

            sample_sen = sentiment_to_onehot(int(sample["sentiment"]), self.sentiment_num_classes)

            full_wave = read_audio(sample_audio_path, self.sr)
            if full_wave is None or full_wave.numel() == 0:
                print(f"File {sample_audio_path} in {self.corpus_name}/{self.subset_name} is missing or empty.")
                continue

            audio_windows = slice_audio(
                start_time=int(sample["start_time"] * self.sr) if "CMUMOSEI" in self.corpus_name else 0,
                end_time=int(sample["end_time"] * self.sr) if "CMUMOSEI" in self.corpus_name else len(full_wave),
                win_max_length=int(self.win_max_length * self.sr),
                win_shift=int(self.win_shift * self.sr),
                win_min_length=int(self.win_min_length * self.sr),
            )

            if not audio_windows:
                continue

            sample_start = 0
            sample_end = (
                int((float(sample["end_time"]) - float(sample["start_time"])) * self.sr)
                if "CMUMOSEI" in self.corpus_name
                else len(full_wave)
            )

            record = {
                "audio_path": str(sample_audio_path),
                "video_path": str(sample_video_path),
                "filename": str(sample_filename),
                "start": sample_start,
                "end": sample_end,
                "target_emo": sample_emo,
                "target_sen": sample_sen,
                "w_idx": 0,
                "data_available": None,
            }

            for modality, fe in self.feature_extractors.items():
                feature_path = self.full_features_paths[modality] / sample_filename.replace(".wav", "_0.dat")
                if not feature_path.exists():
                    save_pickle(
                        {
                            "start": record["start"],
                            "end": record["end"],
                            "target_emo": record["target_emo"],
                            "target_sen": record["target_sen"],
                            "features": self._extract_mean_features(
                                modality=modality,
                                extractor=fe,
                                sample=sample,
                                video_path=sample_video_path,
                                full_wave=full_wave,
                                audio_windows=audio_windows,
                            ),
                            "data_available": None,
                        },
                        feature_path,
                    )

                info[modality].append(record.copy())

        return info

    def _extract_mean_features(
        self,
        *,
        modality: str,
        extractor: Any,
        sample: dict[str, Any],
        video_path: Path,
        full_wave: torch.Tensor,
        audio_windows: list[dict[str, int]],
    ) -> torch.Tensor:
        if modality.lower() == "t":
            text = sample["ASR"] if "RAMAS" in self.corpus_name else sample["text"]
            return torch.as_tensor(extractor(text), dtype=torch.float32)

        features = []
        for window in audio_windows:
            if modality.lower() == "a":
                payload = full_wave[window["start"] : window["end"]].clone()
            elif modality.lower() == "v":
                offset_sec = float(sample["start_time"]) if "CMUMOSEI" in self.corpus_name else 0.0
                payload = {
                    "video_path": str(video_path),
                    "start_sec": float(window["start"]) / self.sr - offset_sec,
                    "end_sec": float(window["end"]) / self.sr - offset_sec,
                }
            else:
                raise NotImplementedError(f"Unsupported modality: {modality}")

            features.append(torch.as_tensor(extractor(payload), dtype=torch.float32))

        return torch.stack(features, dim=0).mean(dim=0)

    def _filter_samples(self, info: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        stats = {
            "db": self.corpus_name,
            "fns": {},
            "majority_class": {"emo": 0, "emo_6": 0, "emo_7": 0, "sen": 0},
            "counts": {
                "emo": np.zeros(self.emotion_num_classes, dtype=np.float64),
                "emo_6": np.zeros(6, dtype=np.float64),
                "emo_7": np.zeros(7, dtype=np.float64),
                "sen": np.zeros(self.sentiment_num_classes, dtype=np.float64),
            },
        }

        filtered = []
        for record in info:
            if self.load_in_ram:
                features = self.load_features(record["filename"], record["w_idx"])
            else:
                features = {f"{modality}_features": None for modality in self.full_features_paths}

            record.update(features)
            filtered.append(record)
            emo_target = np.asarray(record["target_emo"], dtype=np.float32)
            emo_target_6 = np.asarray(emotion_to_target(emo_target, 6), dtype=np.float32)
            emo_target_7 = np.asarray(emotion_to_target(emo_target, 7), dtype=np.float32)
            sen_target = np.asarray(record["target_sen"], dtype=np.float32)
            stats["fns"][record["filename"]] = {
                "emo": emo_target,
                "emo_6": emo_target_6,
                "emo_7": emo_target_7,
                "sen": sen_target,
            }

            stats["counts"]["emo"] += emo_target.astype(np.float64)
            stats["counts"]["emo_6"] += emo_target_6.astype(np.float64)
            stats["counts"]["emo_7"] += emo_target_7.astype(np.float64)
            stats["counts"]["sen"] += sen_target.astype(np.float64)

        stats["majority_class"]["emo"] = int(np.argmax(stats["counts"]["emo"])) if filtered else 0
        stats["majority_class"]["emo_6"] = int(np.argmax(stats["counts"]["emo_6"])) if filtered else 0
        stats["majority_class"]["emo_7"] = int(np.argmax(stats["counts"]["emo_7"])) if filtered else 0
        stats["majority_class"]["sen"] = int(np.argmax(stats["counts"]["sen"])) if filtered else 0
        return filtered, stats

    def load_features(self, filename: str, w_idx: int) -> dict[str, torch.Tensor]:
        res = {}
        features_filename = normalize_audio_filename(filename).replace(".wav", f"_{w_idx}.dat")
        for modality, full_features_paths in self.full_features_paths.items():
            fp = Path(full_features_paths) / features_filename
            if fp.exists():
                m_data = load_pickle(str(fp))
                res.update({f"{modality}_features": torch.FloatTensor(m_data["features"])})
            else:
                res.update({f"{modality}_features": torch.zeros(*self.feature_extractors[modality].output_shape)})

        return res

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.info[index]

        if not self.load_in_ram:
            features = self.load_features(data["filename"], data["w_idx"])
        else:
            features = {key: value.clone() for key, value in data.items() if "features" in key}

        if self.transform is not None:
            features = self.transform(features)

        target = torch.cat(
            (
                torch.as_tensor(data["target_emo"], dtype=torch.float32),
                torch.as_tensor(data["target_sen"], dtype=torch.float32),
            ),
            dim=0,
        )

        return {
            "inputs": features,
            "target": target,
            "meta": {
                "filename": data["filename"],
                "start_t": data["start"] / self.sr,
                "end_t": data["end"] / self.sr,
                "start_f": data["start"],
                "end_f": data["end"],
                "corpus_name": self.corpus_name,
            },
        }
