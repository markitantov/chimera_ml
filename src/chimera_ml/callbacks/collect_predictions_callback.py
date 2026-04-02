import csv
import io
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from chimera_ml.callbacks._utils import resolve_splits
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


@dataclass
class CollectPredictionsCallback(BaseCallback):
    """Log per-sample predictions as CSV artifacts to MLflow.

    `splits` supports:
      - "val"  -> all validation splits (val_loaders dict, else "val")
      - "test" -> all test splits (test_loaders dict, else "test")
      - "train" -> train split
      - any other string -> exact split name (searched in val/test loader dicts)
    """

    splits: list[str] = field(default_factory=lambda: ["val"])
    artifact_path: str = "predictions"
    filename_template: str = "preds_epoch_{epoch}.csv"
    include_probs: bool = True
    task: Literal["regression", "classification"] = "regression"

    def on_fit_start(self, trainer: Any) -> None:
        """Ensure prediction cache collection is enabled for validation/evaluation."""
        trainer.config.collect_cache = True

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Export cached predictions to CSV and attach them as MLflow artifacts."""
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is None:
            return

        for split_name, loader in self._resolve_splits(trainer):
            cached = trainer.get_cached_split_outputs(split_name)
            if cached is None:
                if loader is None:
                    continue

                predict_fn = getattr(trainer, "predict", None)
                if not callable(predict_fn):
                    self._warning(
                        trainer,
                        f"[MLflowPredictionsCallback] No cached predictions for split '{split_name}' "
                        "and trainer.predict(...) is not available. Skipping export.",
                    )
                    continue

                cached = predict_fn(loader, split=split_name)

            preds = getattr(cached, "preds", None)
            if preds is None or self._numel(preds) == 0:
                continue

            targets = getattr(cached, "targets", None)
            metas = getattr(cached, "sample_meta", None)

            n = self._batch_size(preds)
            ids = self._extract_ids(metas, n)

            rows = self._build_rows(preds, targets, ids)
            if not rows:
                continue

            data = self._rows_to_csv_bytes(rows)
            logger.log_artifact_bytes(
                data,
                artifact_path=f"{self.artifact_path}/{split_name}",
                filename=self.filename_template.format(epoch=epoch),
            )

    def _resolve_splits(self, trainer: Any) -> list[tuple[str, DataLoader | None]]:
        """Resolve configured split selectors into concrete (name, loader) pairs."""
        return resolve_splits(trainer, self.splits)

    @staticmethod
    def _numel(x: torch.Tensor | list[torch.Tensor]) -> int:
        """Return total number of stored elements for tensor or ragged chunks."""
        if torch.is_tensor(x):
            return int(x.numel())

        return int(sum(int(t.numel()) for t in x))

    @staticmethod
    def _batch_size(x: torch.Tensor | list[torch.Tensor]) -> int:
        """Return total batch size across tensor or ragged chunks."""
        if torch.is_tensor(x):
            return int(x.shape[0])

        return int(sum(int(t.shape[0]) for t in x))

    @staticmethod
    def _iter_samples(x: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """Flatten cache container into a list of per-sample tensors."""
        if torch.is_tensor(x):
            return [x[i] for i in range(int(x.shape[0]))]

        out: list[torch.Tensor] = []
        for chunk in x:
            out.extend(chunk[i] for i in range(int(chunk.shape[0])))

        return out

    @staticmethod
    def _extract_ids(metas: Any, n: int) -> list[str | None] | None:
        """Extract optional sample IDs from cached metadata."""
        if isinstance(metas, list):
            return [m.get("id") if isinstance(m, dict) else None for m in metas[:n]]

        return None

    def _build_rows(
        self,
        preds: torch.Tensor | list[torch.Tensor],
        targets: torch.Tensor | list[torch.Tensor] | None,
        ids: list[str | None] | None,
    ) -> list[dict[str, object]]:
        """Build CSV-ready row dictionaries for regression/classification tasks."""
        pred_samples = self._iter_samples(preds)
        target_samples = self._iter_samples(targets) if targets is not None else None
        n = len(pred_samples)

        if self.task == "classification":
            rows = []
            for i in range(n):
                logits = pred_samples[i].view(-1)
                pred_class = int(torch.argmax(logits).item())
                row = {
                    "id": ids[i] if ids else None,
                    "pred_class": pred_class,
                    "target": int(target_samples[i].item())
                    if target_samples is not None and target_samples[i].numel() == 1
                    else None,
                }
                if self.include_probs:
                    probs = torch.softmax(logits, dim=-1)
                    for c in range(int(probs.shape[0])):
                        row[f"prob_{c}"] = float(probs[c].item())

                rows.append(row)

            return rows

        rows = []
        for i in range(n):
            pred_flat = pred_samples[i].reshape(-1)
            row = {"id": ids[i] if ids else None}
            for j in range(int(pred_flat.shape[0])):
                row[f"pred_{j}"] = float(pred_flat[j].item())

            if target_samples is not None:
                targ_flat = target_samples[i].reshape(-1)
                for j in range(int(targ_flat.shape[0])):
                    row[f"target_{j}"] = float(targ_flat[j].item())

            rows.append(row)

        return rows

    @staticmethod
    def _rows_to_csv_bytes(rows: list[dict[str, object]]) -> bytes:
        """Serialize row dictionaries into UTF-8 CSV bytes."""
        buf = io.StringIO()
        fieldnames = sorted({k for row in rows for k in row})
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue().encode("utf-8")


@CALLBACKS.register("collect_predictions_callback")
def collect_predictions_callback(**params):
    """Registry factory for :class:`CollectPredictionsCallback`."""
    return CollectPredictionsCallback(**params)
