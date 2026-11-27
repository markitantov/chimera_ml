import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix for arbitrary label values."""
    labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_idx = {label: idx for idx, label in enumerate(labels.tolist())}

    true_idx = np.asarray([label_to_idx[v] for v in y_true], dtype=np.int64)
    pred_idx = np.asarray([label_to_idx[v] for v in y_pred], dtype=np.int64)

    cm = np.zeros((labels.size, labels.size), dtype=np.int64)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def normalize_confusion_matrix(cm: np.ndarray, normalize: str) -> np.ndarray:
    """Normalize confusion matrix like sklearn: true/pred/all."""
    cmf = cm.astype(np.float64, copy=True)

    if normalize == "true":
        row_sums = cmf.sum(axis=1, keepdims=True)
        return np.divide(cmf, row_sums, out=np.zeros_like(cmf), where=row_sums != 0.0)

    if normalize == "pred":
        col_sums = cmf.sum(axis=0, keepdims=True)
        return np.divide(cmf, col_sums, out=np.zeros_like(cmf), where=col_sums != 0.0)

    if normalize == "all":
        total = float(cmf.sum())
        if total == 0.0:
            return np.zeros_like(cmf)
        return cmf / total

    raise ValueError("normalize must be one of: None, 'true', 'pred', 'all'")
