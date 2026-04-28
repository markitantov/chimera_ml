from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from common.utils import (
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
from torch.utils.data import Dataset
from tqdm import tqdm


class AGenderAudioDataset(Dataset):
    def __init__(
        self,
        *,
        data_root: str,
        labels_metadata: pd.DataFrame,
        features_root: str,
        features_file_name: str,
        corpus_name: str,
        gender_num_classes: int,
        vad_metadata: dict[str, list[dict[str, int]]] | None = None,
        sr: int = 16000,
        win_max_length: int = 4,
        win_shift: int = 2,
        win_min_length: int = 0,
        transform: Any = None,
        preprocessor_name: str | None = None,
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
        self.data_preprocessor = None
        if preprocessor_name:
            from common.data_preprocessors import Wav2Vec2DataPreprocessor

            self.data_preprocessor = Wav2Vec2DataPreprocessor(
                preprocessor_name=preprocessor_name,
                sr=self.sr,
                win_max_length=self.win_max_length,
            )

        suffix = generate_features_suffix(
            vad_metadata=self.vad_metadata,
            win_max_length=self.win_max_length,
            win_shift=self.win_shift,
            win_min_length=self.win_min_length,
        )

        self.full_features_path = Path(features_root) / f"{features_file_name}_{suffix}"
        self.full_features_path.mkdir(parents=True, exist_ok=True)
        stats_path = Path(features_root) / f"{features_file_name}_{suffix}_stats.pickle"

        self.display_filtering_stats = display_filtering_stats

        info = load_pickle(stats_path)

        if not info:
            print("No cached feature index found. Start reading audio and preparing window caches.")
            info = self._prepare_data()
            save_pickle(info, stats_path)
            print(f"Feature index saved to '{stats_path}'.")
        else:
            print(f"Using cached feature index from '{stats_path}'.")
        
        self.info, self.stats = self._filter_samples(info)
        print(
            f"Ready: num_windows={len(self.info)}, "
            f"num_files={len(self.stats['fns'])}, "
            f"gender_counts={self.stats['counts']['gen'].tolist()}"
        )

    def _prepare_data(self) -> list[dict[str, Any]]:
        info: list[dict[str, Any]] = []
        records = self.labels_metadata.to_dict("records")
        files_without_windows_before_vad = 0
        files_without_windows_after_vad = 0
        windows_before_vad = 0
        windows_after_vad = 0

        for sample in tqdm(records, desc=f"{self.corpus_name}: extracting audio features"):
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

            for w_idx, window in enumerate(windows):
                wave = full_wave[window["start"] : window["end"]].clone()
                info.append(
                    {
                        "fp": str(sample_fp),
                        "fn": sample_filename,
                        "w_idx": w_idx,
                        "start": window["start"],
                        "end": window["end"],
                        "gen": sample["gender"],
                        "age": sample["age"],
                        "mask": "No mask",
                    }
                )

                save_pickle(wave, self.full_features_path / waveform_cache_name(sample_filename, w_idx))

        if self.display_filtering_stats:
            print(
                f"Prepared {len(info)} window records from {len(records)} source rows. "
                f"files_without_windows_before_vad={files_without_windows_before_vad}, "
                f"files_without_windows_after_vad={files_without_windows_after_vad}, "
                f"windows_before_vad={windows_before_vad}, "
                f"windows_after_vad={windows_after_vad}"
            )
        
        return info

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
        audio = load_pickle(self.full_features_path / waveform_cache_name(data["fn"], data["w_idx"]))
        if audio is None:
            raise FileNotFoundError(self.full_features_path / waveform_cache_name(data["fn"], data["w_idx"]))
    
        if self.transform:
            audio = self.transform(audio)

        if self.data_preprocessor:
            audio = self.data_preprocessor(audio)
        else:
            target_len = int(self.win_max_length * self.sr)
            audio = torch.nn.functional.pad(audio, (0, max(0, target_len - len(audio))), mode="constant")[:target_len]

        target = torch.tensor([float(data["gen"]), float(data["age"])], dtype=torch.float32)
        if torch.is_tensor(audio):
            audio_tensor = audio.to(dtype=torch.float32)
        else:
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return {
            "inputs": {"audio": audio_tensor},
            "target": target,
            "meta": {
                "filename": data["fn"],
                "start_t": data["start"] / self.sr,
                "end_t": data["end"] / self.sr,
                "start_f": data["start"],
                "end_f": data["end"],
                "corpus_name": self.corpus_name,
            },
        }
