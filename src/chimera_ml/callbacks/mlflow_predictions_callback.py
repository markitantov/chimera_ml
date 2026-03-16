from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Literal

import csv
import io
import torch
from torch.utils.data import DataLoader

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


# TODO
@dataclass
class MLflowPredictionsCallback(BaseCallback):
    """Log per-sample predictions as CSV artifact to MLflow.

    `splits` is a list of split selectors:
      - "val"  -> all validation splits (val_loaders dict, else "val")
      - "test" -> all test splits (test_loaders dict, else "test")
      - "train"-> train
      - any other string -> exact split name (searched in val/test loader dicts)
    """
    splits: List[str] = field(default_factory=lambda: ["val"])
    artifact_path: str = "predictions"
    filename_template: str = "preds_epoch_{epoch}.csv"
    include_probs: bool = True
    task: Literal["regression", "classification"] = "regression"

    def on_fit_start(self, trainer) -> None:
        # Ensure predictions are collected during validation so callbacks don't rerun forward passes.
        trainer.config.collect_cache = True

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, float]) -> None:
        logger = getattr(trainer, "mlflow_logger", None)
        if logger is None:
            return

        for split_name, loader in self._resolve_splits(trainer):
            cached = trainer.get_cached_predictions(split_name)
            if cached is None:
                if loader is None:
                    continue
                cached = trainer.predict(loader, split=split_name)

            preds = getattr(cached, "preds", None)
            if preds is None or preds.numel() == 0:
                continue

            targets = getattr(cached, "targets", None)
            metas = getattr(cached, "sample_meta", None)

            n = int(preds.shape[0])
            ids = self._extract_ids(metas, n)

            rows = self._build_rows(preds[:n], targets[:n] if targets is not None else None, ids)
            if not rows:
                continue

            data = self._rows_to_csv_bytes(rows)
            logger.log_artifact_bytes(
                data,
                artifact_path=f"{self.artifact_path}/{split_name}",
                filename=self.filename_template.format(epoch=epoch),
            )

    def _resolve_splits(self, trainer) -> List[Tuple[str, Optional[DataLoader]]]:
        out: List[Tuple[str, Optional[DataLoader]]] = []
        seen: set[str] = set()

        def add(name: str, loader: Optional[DataLoader]) -> None:
            if name and name not in seen:
                seen.add(name)
                out.append((name, loader))

        val_loaders = getattr(trainer, "_val_loaders", None)
        test_loaders = getattr(trainer, "_test_loaders", None)

        for s in self.splits or ["val"]:
            if s == "train":
                add("train", getattr(trainer, "_train_loader", None))
                continue

            if s == "val":
                if isinstance(val_loaders, dict) and val_loaders:
                    for k, v in val_loaders.items():
                        add(k, v)
                else:
                    add("val", getattr(trainer, "_val_loader", None))
                continue

            if s == "test":
                if isinstance(test_loaders, dict) and test_loaders:
                    for k, v in test_loaders.items():
                        add(k, v)
                else:
                    add("test", getattr(trainer, "_test_loader", None))
                continue

            # exact split name
            loader = None
            if isinstance(val_loaders, dict) and s in val_loaders:
                loader = val_loaders[s]
            elif isinstance(test_loaders, dict) and s in test_loaders:
                loader = test_loaders[s]
            add(s, loader)

        return out

    @staticmethod
    def _extract_ids(metas, n: int) -> Optional[List[Optional[str]]]:
        if isinstance(metas, list):
            return [m.get("id") if isinstance(m, dict) else None for m in metas[:n]]
        return None

    def _build_rows(self, preds: torch.Tensor, targets: Optional[torch.Tensor], ids):
        n = int(preds.shape[0])

        # --- CLASSIFICATION ---
        if self.task == "classification":
            if preds.ndim != 2:
                preds = preds.view(n, -1)
            logits = preds
            pred_class = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1) if self.include_probs else None

            rows = []
            for i in range(n):
                row = {
                    "id": ids[i] if ids else None,
                    "pred_class": int(pred_class[i].item()),
                    "target": int(targets[i].item()) if targets is not None and targets[i].numel() == 1 else None,
                }
                if probs is not None:
                    for c in range(int(probs.shape[1])):
                        row[f"prob_{c}"] = float(probs[i, c].item())
                rows.append(row)
            return rows

        # --- REGRESSION (default) ---
        flat = preds.view(n, -1)
        targ_flat = targets.view(n, -1) if targets is not None else None

        rows = []
        for i in range(n):
            row = {"id": ids[i] if ids else None}
            for j in range(int(flat.shape[1])):
                row[f"pred_{j}"] = float(flat[i, j].item())
            if targ_flat is not None:
                for j in range(int(targ_flat.shape[1])):
                    row[f"target_{j}"] = float(targ_flat[i, j].item())
            rows.append(row)
        return rows


    @staticmethod
    def _rows_to_csv_bytes(rows: List[Dict[str, object]]) -> bytes:
        buf = io.StringIO()
        fieldnames = sorted(rows[0].keys())
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue().encode("utf-8")


@CALLBACKS.register("mlflow_predictions_callback")
def mlflow_predictions_callback(**params):
    return MLflowPredictionsCallback(**params)