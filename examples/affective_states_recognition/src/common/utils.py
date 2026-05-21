import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


def compute_class_weights(counts: np.ndarray) -> list[float]:
    counts = np.asarray(counts, dtype=np.float64)
    positive = counts > 0
    if not np.any(positive):
        return [1.0] * len(counts)

    weights = np.ones_like(counts, dtype=np.float64)
    weights[positive] = counts[positive].sum() / (len(counts) * counts[positive])
    return weights.tolist()


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

    if not os.path.exists(filepath):
        return None

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
            if chunk_end == end_time:
                break
        else:
            if end_time - cursor >= win_min_length:
                timings.append({"start": cursor, "end": end_time})
            break

        cursor += win_shift

    return timings


def find_intersections(x: list[dict[str, int]], y: list[dict[str, int]], min_length: int = 0) -> list[dict[str, int]]:
    timings: list[dict[str, int]] = []
    i = j = 0
    while i < len(x) and j < len(y):
        left = max(int(x[i]["start"]), int(y[j]["start"]))
        right = min(int(x[i]["end"]), int(y[j]["end"]))
        if left <= right and right - left >= min_length:
            timings.append({"start": left, "end": right})

        if x[i]["end"] < y[j]["end"]:
            i += 1
        else:
            j += 1

    return timings


def emotion_to_target(value: list[float] | np.ndarray, num_emotions: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] == num_emotions:
        return arr

    if num_emotions == 6 and arr.shape[0] == 7:
        return arr[1:]

    if num_emotions == 7 and arr.shape[0] >= 7:
        return arr[:7]

    raise ValueError(f"Cannot convert emotion target with shape {arr.shape} to {num_emotions} classes.")


def sentiment_to_index(value: int, num_classes: int) -> int:
    return {
        -1: 0,
        0: 1 if num_classes == 3 else -1,
        1: 2 if num_classes == 3 else 1,
    }[int(value)]


def sentiment_to_onehot(value: int, num_classes: int) -> np.ndarray | None:
    index = sentiment_to_index(value, num_classes)
    if index < 0:
        return None

    return (np.arange(num_classes) == index).astype(np.float32)


def mode_label(values: list[int] | np.ndarray) -> int:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return 0

    arr = arr[arr >= 0]
    if arr.size == 0:
        return 0

    return int(np.bincount(arr).argmax())


def majority_multilabel(values: list[np.ndarray] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    return (arr.sum(axis=0) > (arr.shape[0] / 2.0)).astype(np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    labels = np.unique(np.concatenate((y_true, y_pred), axis=0))
    labels = labels[labels >= 0]
    if labels.size == 0:
        return {"war": 0.0, "uar": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0}

    label_to_index = {int(label): idx for idx, label in enumerate(labels.tolist())}
    cm = np.zeros((labels.size, labels.size), dtype=np.float64)
    for target, pred in zip(y_true.tolist(), y_pred.tolist(), strict=False):
        if int(target) in label_to_index and int(pred) in label_to_index:
            cm[label_to_index[int(target)], label_to_index[int(pred)]] += 1.0

    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    present = support > 0
    total = float(np.sum(support))
    weights = np.divide(support, total, out=np.zeros_like(support), where=total != 0.0)
    return {
        "war": float(np.sum(weights * recall) * 100.0),
        "uar": float(np.mean(recall[present]) * 100.0) if np.any(present) else 0.0,
        "weighted_f1": float(np.sum(weights * f1) * 100.0),
        "macro_f1": float(np.mean(f1[present]) * 100.0) if np.any(present) else 0.0,
    }


def cmu_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_label = np.asarray(y_true) > 0
    predicted_label = np.asarray(y_pred) > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))
    return (tp * (n / (p + 1e-16)) + tn) / (2 * n + 1e-16)


def cmu_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = [str(name) for name in (class_names or [])]

    # CMU-MOSEI multilabel metrics are defined over the 6 non-neutral emotions.
    if y_true.ndim == 2 and y_true.shape[1] > 6:
        y_true = y_true[:, 1:7]

    if y_pred.ndim == 2 and y_pred.shape[1] > 6:
        y_pred = y_pred[:, 1:7]

    if len(labels) > 6:
        labels = labels[1:7]

    per_class_metrics: dict[str, dict[str, float]] = {
        "mWA": {},
        "mWF1": {},
        "mMacroF1": {},
        "mA_WAR": {},
    }

    if y_true.size == 0 or y_pred.size == 0:
        aggregate_metrics = {
            "emo_mWA": 0.0,
            "emo_mWF1": 0.0,
            "emo_mMacroF1": 0.0,
            "emo_mA_WAR": 0.0,
        }

        for label in labels:
            per_class_metrics["mWA"][label] = 0.0
            per_class_metrics["mWF1"][label] = 0.0
            per_class_metrics["mMacroF1"][label] = 0.0
            per_class_metrics["mA_WAR"][label] = 0.0

        return aggregate_metrics, per_class_metrics

    mwa, mwf1, mmacrof1, mwar = [], [], [], []
    for index in range(y_pred.shape[1]):
        target_col = y_true[:, index]
        pred_col = y_pred[:, index]
        cls = classification_metrics(target_col, pred_col)
        class_mwa = cmu_accuracy(target_col, pred_col) * 100.0
        class_mwf1 = cls["weighted_f1"]
        class_mmacrof1 = cls["macro_f1"]
        class_mwar = cls["war"]
        mwa.append(class_mwa)
        mwf1.append(class_mwf1)
        mmacrof1.append(class_mmacrof1)
        mwar.append(class_mwar)

        if index < len(labels):
            label = labels[index]
            per_class_metrics["mWA"][label] = float(class_mwa)
            per_class_metrics["mWF1"][label] = float(class_mwf1)
            per_class_metrics["mMacroF1"][label] = float(class_mmacrof1)
            per_class_metrics["mA_WAR"][label] = float(class_mwar)

    aggregate_metrics = {
        "emo_mWA": float(np.mean(mwa)),
        "emo_mWF1": float(np.mean(mwf1)),
        "emo_mMacroF1": float(np.mean(mmacrof1)),
        "emo_mA_WAR": float(np.mean(mwar)),
    }

    return aggregate_metrics, per_class_metrics


def generate_features_suffix(
    *,
    vad_metadata: Any,
    win_max_length: int,
    win_shift: int,
    win_min_length: int,
) -> str:
    return f"{'VAD' if vad_metadata else ''}{win_max_length}{win_shift}{win_min_length}"


def normalize_audio_filename(value: str) -> str:
    temp = str(value).replace(".mp3", ".wav").replace(".m4a", ".wav").replace(".mov", ".wav")
    if ".wav" in temp:
        return temp

    return f"{temp}.wav"
