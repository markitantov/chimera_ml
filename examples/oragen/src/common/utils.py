import os
import pickle
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

import PIL


class DatasetType(IntEnum):
    AUDIO = 1
    VIDEO = 2
    BOTH = 3


class FeaturesType(IntEnum):
    EARLY = 1
    INTERMEDIATE = 2
    LATE = 3


def load_pickle(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_audio(filepath: str | Path, sample_rate: int) -> torch.Tensor:
    import torchaudio

    wave, sr = torchaudio.load(str(filepath))
    if wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    
    if sr != sample_rate:
        wave = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wave)
    
    return wave.squeeze(0)


def slice_audio(
    *,
    start_time: int,
    end_time: int,
    win_max_length: int,
    win_shift: int,
    win_min_length: int,
) -> list[dict[str, int]]:
    if end_time < start_time:
        return []

    if (end_time - start_time) <= win_max_length:
        return [{"start": start_time, "end": end_time}]

    timings: list[dict[str, int]] = []
    cursor = start_time
    while cursor < end_time:
        chunk_end = cursor + win_max_length
        if chunk_end <= end_time:
            timings.append({"start": cursor, "end": chunk_end})
            # if tail exact `win_max_length` seconds
            if chunk_end == end_time: 
                break

        else: # if tail less then `win_max_length` seconds
            if end_time - cursor >= win_min_length:
                timings.append({"start": cursor, "end": end_time})
            
            break
        
        cursor += win_shift
    
    return timings


def find_intersections(x: list[dict[str, int]], y: list[dict[str, int]], min_length: int = 0) -> list[dict[str, int]]:
    timings: list[dict[str, int]] = []
    i = j = 0
    while i < len(x) and j < len(y):
        # Left bound for intersecting segment
        left = max(int(x[i]["start"]), int(y[j]["start"]))

        # Right bound for intersecting segment
        right = min(int(x[i]["end"]), int(y[j]["end"]))

        # If segment is valid and 
        # length of intersection not less then `min_length` seconds
        if left <= right and right - left >= min_length: 
            timings.append({"start": left, "end": right})
        
        # If i-th interval's right bound is 
        # smaller increment i else increment j
        if x[i]["end"] < y[j]["end"]:
            i += 1
        else:
            j += 1
    
    return timings


def generate_features_suffix(
    *,
    vad_metadata: Any,
    win_max_length: int,
    win_shift: int,
    win_min_length: int,
) -> str:
    return f"{'VAD' if vad_metadata else ''}{win_max_length}{win_shift}{win_min_length}"


def gender_label_to_int(value: str, num_classes: int) -> int:
    """Convert gender value to label
    Child -> 0 class
    Female -> 1 class
    Male -> 2 class

    Args:
        value (str): Gender label
        num_classes (str): Number of classes: binary or ternary problem

    Returns:
        int: Converted Gender label
    """
    value = str(value).strip().lower()
    mapping = {
        2: {"female": 0, "male": 1},
        3: {"child": 0, "female": 1, "male": 2},
    }

    return mapping[num_classes][value]


def mask_label_to_int(value: str) -> int:
    """Convert mask value to label
    No mask -> 0 class
    Tissue mask -> 1 class
    Medical mask -> 2 class
    Protective mask (ffp2/ffp3) -> 3 class
    Respirator -> 4 class
    Protective face shield -> 5 class

    Args:
        value (str): Mask label value

    Returns:
        int: Converted Mask label
    """
    return {
        "No mask": 0,
        "Tissue mask": 1,
        "Medical mask": 2,
        "Protective mask (ffp2/ffp3)": 3,
        "Respirator": 4,
        "Protective face shield": 5,
    }[str(value)]


def normalize_audio_filename(value: str) -> str:
    return str(value).replace(".mp3", ".wav").replace(".m4a", ".wav")


def waveform_cache_name(filename: str, window_index: int) -> str:
    return normalize_audio_filename(filename).replace(".wav", f"_{window_index}.dat")


def ensure_existing_file(path: str | Path, *, hint: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{hint}: {path}")


def define_context_length(win_max_length: int = 4) -> int:
    return {
        1: 49,
        2: 99,
        3: 149,
        4: 199
    }[win_max_length]
    
    
def read_img(path: str):
    img = PIL.Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img


def multitask_dict_to_tensor(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [outputs["gen"], outputs["age"].unsqueeze(-1)]
    if "mask" in outputs:
        parts.append(outputs["mask"])
    
    return torch.cat(parts, dim=-1)

