from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Mapping, Tuple, Union, Callable
import logging
import re
import inspect

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from chimera_ml.core.batch import Batch
from chimera_ml.models.base import BaseModel
from chimera_ml.losses.base import BaseLoss
from chimera_ml.metrics.base import BaseMetric
from chimera_ml.logging.base import BaseLogger
from chimera_ml.metrics.sklearn_confusion_matrix import SklearnConfusionMatrixMetric
from chimera_ml.visualization.confusion_matrix import plot_confusion_matrix, fig_to_png_bytes
from chimera_ml.training.config import TrainConfig
from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.training.predictions_store import EpochPredictions


FeatureExtractor = Callable[[Any, Any, Any], torch.Tensor] # TODO

@dataclass
class Trainer:
    model: BaseModel
    loss_fn: BaseLoss
    optimizer: torch.optim.Optimizer
    metrics: List[BaseMetric]
    config: TrainConfig
    mlflow_logger: Optional[BaseLogger] = None
    logger: Optional[logging.Logger] = None
    class_names: Optional[List[str]] = None
    callbacks: List[BaseCallback] = field(default_factory=list)
    scheduler: Optional[Any] = None
    stop_training: bool = False

    global_step: int = 0
    predictions_cache: Dict[str, EpochPredictions] = field(default_factory=dict)

    def fit(
        self,
        train_loaders: Union[DataLoader, Mapping[str, DataLoader], List[DataLoader], Tuple[DataLoader, ...]],
        val_loaders: Optional[Union[DataLoader, Mapping[str, DataLoader], List[DataLoader], Tuple[DataLoader, ...]]] = None,
    ) -> None:
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Expose loaders for callbacks (e.g., predictions logger)
        self._train_loaders = self._normalize_loaders(train_loaders, default_name="train")
        self._train_loader = next(iter(self._train_loaders.values()), None) # TODO
        self._val_loaders = self._normalize_loaders(val_loaders, default_name="val")
        # Backwards compat for callbacks that look for _val_loader
        self._val_loader = next(iter(self._val_loaders.values()), None)
        self._test_loaders = {}
        self._test_loader = None

        use_amp = bool(self.config.mixed_precision) and (device.type == "cuda")
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        if self.mlflow_logger:
            params = {
                "mode": "train",
                "epochs": self.config.epochs,
                "mixed_precision": self.config.mixed_precision,
                "use_scheduler": self.config.use_scheduler,
                "scheduler_step_per_epoch": self.config.scheduler_step_per_epoch,
                "collect_cache": self.config.collect_cache,
            }

            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("name", f"group{i}")
                params[f"optimizer_lr/{name}"] = float(pg.get("lr", 0.0))
                params[f"optimizer_weight_decay/{name}"] = float(pg.get("weight_decay", 0.0))
            
            self.mlflow_logger.start(params=params)

        for cb in self.callbacks:
            cb.on_fit_start(self)
        
        if self.logger:
            self.logger.info("🚀 Run %s started.", self.logger.run_name)
            self.logger.info("📸 Logging to: %s", self.logger.log_path)
        else:
            print("🚀 Run %s started.")

        self.stop_training = False
        self.global_step = 0

        for epoch in range(1, self.config.epochs + 1):
            if self.stop_training:
                break

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            train_logs = self._run_train_epoch(
                loaders=self._train_loaders,
                device=device,
                scaler=scaler,
                epoch=epoch,
            )

            if self.mlflow_logger:
                self.mlflow_logger.log_metrics({f"train/{k}": v for k, v in train_logs.items()}, step=epoch)

            logs_for_callbacks = {f"train/{k}": v for k, v in train_logs.items()}

            if self._val_loaders:
                # primary split (for backwards-compatible val/* monitors)
                primary_split = "val" if "val" in self._val_loaders else next(iter(self._val_loaders.keys()))

                for split_name, loader in self._val_loaders.items():
                    val_logs = self._run_epoch(
                        loader=loader,
                        device=device,
                        scaler=scaler,
                        train=False,
                        epoch=epoch,
                        split=split_name,
                    )
                    if self.mlflow_logger:
                        self.mlflow_logger.log_metrics({f"{split_name}/{k}": v for k, v in val_logs.items()}, step=epoch)

                    logs_for_callbacks.update({f"{split_name}/{k}": v for k, v in val_logs.items()})

                    # Confusion matrix (if metric is present) for this split
                    cm_metric = next((m for m in self.metrics if isinstance(m, SklearnConfusionMatrixMetric)), None)
                    if cm_metric is not None:
                        cm = cm_metric.value()
                        if isinstance(cm, np.ndarray) and cm.ndim == 2 and cm.shape[0] == cm.shape[1]:
                            fig = plot_confusion_matrix(
                                cm,
                                labels=self.class_names,
                                title=f"{split_name} Confusion Matrix (epoch {epoch})",
                            )

                            try:
                                png = fig_to_png_bytes(fig)
                                if self.mlflow_logger:
                                    self.mlflow_logger.log_artifact_bytes(
                                        png,
                                        artifact_path=f"figures/{split_name}",
                                        filename=f"confusion_matrix_epoch_{epoch}.png",
                                    )
                            finally:
                                plt.close(fig)

                    # Backwards-compat alias: if monitors still use val/* but primary split isn't named 'val'
                    if split_name == primary_split and split_name != "val":
                        for k, v in val_logs.items():
                            logs_for_callbacks.setdefault(f"val/{k}", v)

            if self.mlflow_logger:
                for g in self.optimizer.param_groups:
                    name = g.get("name", "group")

                    to_log = {f"lr/{name}": v for k, v in g.items() if k == "lr"}
                    self.mlflow_logger.log_metrics(to_log, step=epoch)

                    logs_for_callbacks[f"lr/{name}"] = float(g.get("lr", 0.0))

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs_for_callbacks)

            if self.config.use_scheduler and self.config.scheduler_step_per_epoch:
                self._scheduler_step(logs_for_callbacks)

            if self.logger:
                self.logger.info("-" * 100)
            else:
                print("-" * 100)

        for cb in self.callbacks:
            cb.on_fit_end(self)

        if self.mlflow_logger:
            self.mlflow_logger.end()

        if self.logger:
            self.logger.info("🌋Training complete")
        else:
            print("🌋Training complete")

    @torch.no_grad()
    def evaluate(
        self,
        loaders: Mapping[str, "DataLoader"],
        *,
        with_features: Optional[bool] = False,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> Dict[str, float]:
        """Run evaluation only (no optimization), return computed metrics, predictions, features (optional).

        Note: callbacks are executed (on_fit_start/on_epoch_end/on_fit_end) so that
        validation-only callbacks work in eval mode too.
        """
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        use_amp = bool(self.config.mixed_precision) and (device.type == "cuda")
        scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

        self._loaders = self._normalize_loaders(loaders, default_name="all")

        for cb in self.callbacks:
            cb.on_fit_start(self)
            cb.on_epoch_start(self, 1)

        results = {}

        for split_name, dl in self._loaders.items():
            logs = self._run_epoch(
                loader=dl,
                device=device,
                scaler=scaler,
                train=False,
                epoch=1,
                split=split_name,
                with_features=with_features,
                feature_extractor=feature_extractor,
            )

            results.update({f"{split_name}/{k}": v for k, v in logs.items()})

        for cb in self.callbacks:
            cb.on_epoch_end(self, 1, results)
            cb.on_fit_end(self)    

        return results

    def get_cached_predictions(self, split: str) -> Optional[EpochPredictions]:
        return self.predictions_cache.get(split)
    
    def _extract_features(self, out, batch, feature_extractor) -> torch.Tensor:
        if feature_extractor is not None:
            feats = feature_extractor(self.model, batch, out)
            if feats is None:
                raise ValueError("feature_extractor returned None")
            return feats

        aux = getattr(out, "aux", None)
        if isinstance(aux, dict):
            v = aux.get("features", None)
            if v is not None:
                return v

        raise ValueError("with_features=True, but features were not found...")

    def _run_epoch(
        self,
        loader: DataLoader,
        device: torch.device,
        scaler: torch.amp.GradScaler,
        train: bool,
        epoch: int,
        split: str,
        with_features: Optional[bool] = False,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> Dict[str, float]:
        for m in self.metrics:
            m.reset()

        self.model.train(mode=train)
        running_loss = 0.0
        loss_steps = 0
        num_samples = 0

        # reset cached predictions for this split/epoch
        if not train and self.config.collect_cache:
            self.predictions_cache.pop(split, None)

        collect_preds = (not train) and bool(self.config.collect_cache)

        preds_chunks = []
        targets_chunks = []
        feats_chunks = [] if with_features else None
        sample_meta = []
        seen = 0

        pbar = tqdm(loader, desc=f"{split} epoch {epoch}", leave=False)

        use_amp = bool(self.config.mixed_precision) and (device.type == "cuda")

        for batch_raw in pbar:
            batch = self._to_device(batch_raw, device)

            has_targets = batch.targets is not None
            if train and not has_targets:
                raise ValueError("Training batch has no targets. For inference use Trainer.predict() or evaluate(train=False).")

            with torch.set_grad_enabled(train):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = self.model(batch)
                    loss = self.loss_fn(out, batch) if has_targets else None

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    assert loss is not None
                    scaler.scale(loss).backward()

                    if self.config.grad_clip_norm is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                    self.global_step += 1

                    if self.scheduler is not None and self.config.use_scheduler and not self.config.scheduler_step_per_epoch:
                        self._scheduler_step(None)

            batch_size = int(out.preds.shape[0]) if hasattr(out, "preds") and out.preds is not None else 0
            num_samples += batch_size

            if loss is not None:
                running_loss += float(loss.item())
                loss_steps += 1

            if has_targets:
                for metric in self.metrics:
                    metric.update(out, batch)

            # cache predictions (CPU) for callbacks
            if collect_preds:
                p = out.preds.detach().cpu()
                t = batch.targets.detach().cpu() if batch.targets is not None else None

                take = p.shape[0]
                if take > 0:
                    preds_chunks.append(p[:take])
                    if t is not None:
                        targets_chunks.append(t[:take])

                    if with_features:
                        feats = self._extract_features(out, batch, feature_extractor)
                        feats = feats.detach().cpu()
                        if feats.shape[0] != take:
                            raise ValueError(f"features batch size {feats.shape[0]} != preds batch size {take}")
                        feats_chunks.append(feats[:take])

                    if batch.meta and isinstance(batch.meta.get("sample_meta"), list):
                        sample_meta.extend(batch.meta["sample_meta"][:take])

                    seen += take

            avg_loss = running_loss / max(loss_steps, 1) if loss_steps > 0 else float("nan")
            pbar.set_postfix({"loss": f"{avg_loss:.4f}" if loss_steps > 0 else "-"})

            if train:
                if self.mlflow_logger and (self.global_step % self.config.log_every_steps == 0):
                    self.mlflow_logger.log_metrics({"train_step/loss": avg_loss}, step=self.global_step)

                for cb in self.callbacks:
                    cb.on_batch_end(self, self.global_step, {"train_step/loss": avg_loss})

        metrics_out = {"num_samples": int(num_samples)}
        if loss_steps > 0:
            metrics_out["loss"] = running_loss / max(loss_steps, 1)
        for metric in self.metrics:
            metrics_out.update(metric.compute())

        if collect_preds and preds_chunks:
            preds = torch.cat(preds_chunks, dim=0)
            targets = torch.cat(targets_chunks, dim=0) if targets_chunks else None
            metas = sample_meta if sample_meta else None
            feats = torch.cat(feats_chunks, dim=0) if (with_features and feats_chunks) else None
            self.predictions_cache[split] = EpochPredictions(preds=preds, targets=targets, sample_meta=metas, features=feats)

        if not train:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}"
                for k, v in metrics_out.items()
            )
            
            if self.logger:
                self.logger.info(f"[{split} epoch {epoch}] {metrics_str}")
            else:
                print(f"[{split} epoch {epoch}] {metrics_str}")

        return metrics_out

    def _iter_mixed_train_batches(
        self,
        loaders: Dict[str, DataLoader],
        *,
        mode: str,
        stop_on: str,
        epoch: int,
    ):
        """Yield (loader_name, batch_raw). Only 'single' mode is supported."""
        names = list(loaders.keys())
        if not names:
            return

        mode = (mode or "single").lower()
        if mode != "single":
            raise NotImplementedError(
                f"train_loader_mode='{mode}' is not supported. Only 'single' is implemented."
            )

        name = names[0]
        for b in loaders[name]:
            yield name, b

    def _run_train_epoch(
        self,
        loaders: Dict[str, DataLoader],
        device: torch.device,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with support for multiple train loaders.

        Returns a flat dict of metrics. Per-loader losses are returned as:
            {"<loader_name>/loss": value, "<loader_name>/num_samples": value}
        plus the usual aggregate keys: "loss", "num_samples", and metric.compute() outputs.
        """
        for m in self.metrics:
            m.reset()

        self.model.train(True)

        mode = getattr(self.config, "train_loader_mode", "round_robin")
        stop_on = getattr(self.config, "train_stop_on", "min")

        total_loss = 0.0
        total_steps = 0
        total_samples = 0

        per_loss = {name: 0.0 for name in loaders}
        per_steps = {name: 0 for name in loaders}
        per_samples = {name: 0 for name in loaders}

        mode = getattr(self.config, "train_loader_mode", "single")
        stop_on = getattr(self.config, "train_stop_on", "min")  # not implemented

        first_name = next(iter(loaders.keys()))
        try:
            total_steps_expected = len(loaders[first_name])   # batches per epoch
        except Exception:
            total_steps_expected = None

        pbar = tqdm(
            self._iter_mixed_train_batches(loaders, mode=mode, stop_on=stop_on, epoch=epoch),
            total=total_steps_expected,
            desc=f"train epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        )

        use_amp = bool(self.config.mixed_precision) and (device.type == "cuda")

        for loader_name, batch_raw in pbar:
            batch = self._to_device(batch_raw, device)
            if batch.targets is None:
                raise ValueError("Training batch has no targets. For inference use Trainer.predict() or evaluate(train=False).")

            with torch.set_grad_enabled(True):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = self.model(batch)
                    loss = self.loss_fn(out, batch)
                
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                if self.config.grad_clip_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                scaler.step(self.optimizer)
                scaler.update()
                self.global_step += 1

                if self.scheduler is not None and self.config.use_scheduler and not self.config.scheduler_step_per_epoch:
                    self._scheduler_step(None)

            bs = int(out.preds.shape[0]) if hasattr(out, "preds") and out.preds is not None else 0
            total_samples += bs
            per_samples[loader_name] += bs

            li = float(loss.item())
            total_loss += li
            total_steps += 1
            per_loss[loader_name] += li
            per_steps[loader_name] += 1

            for metric in self.metrics:
                metric.update(out, batch)

            avg_total = total_loss / max(total_steps, 1)
            avg_loader = per_loss[loader_name] / max(per_steps[loader_name], 1)
            pbar.set_postfix({"loss": f"{avg_total:.4f}", "src": loader_name})

            if self.mlflow_logger and (self.global_step % self.config.log_every_steps == 0):
                self.mlflow_logger.log_metrics(
                    {
                        "train_step/loss": avg_total,
                        f"train_step/{loader_name}/loss": avg_loader,
                    },
                    step=self.global_step,
                )

            for cb in self.callbacks:
                cb.on_batch_end(
                    self,
                    self.global_step,
                    {
                        "train_step/loss": avg_total,
                        f"train_step/{loader_name}/loss": avg_loader,
                    },
                )

        metrics_out = {"num_samples": int(total_samples)}
        if total_steps > 0:
            metrics_out["loss"] = total_loss / max(total_steps, 1)

        # per-loader summaries
        for name in loaders:
            if per_steps[name] > 0:
                metrics_out[f"{name}/loss"] = per_loss[name] / max(per_steps[name], 1)
            metrics_out[f"{name}/num_samples"] = int(per_samples[name])

        for metric in self.metrics:
            metrics_out.update(metric.compute())

        metrics_str = " | ".join(
            f"{k}: {v:.4f}"
            for k, v in metrics_out.items()
                if "/" not in k  # optionally skip per-loader summaries like "train0/loss"
        )

        if self.logger:
            self.logger.info(f"[train epoch {epoch}] {metrics_str}")
        else:
            print(f"[train epoch {epoch}] {metrics_str}")

        return metrics_out
    
    def _to_device(self, batch: Batch, device: torch.device) -> Batch:
        inputs = {k: v.to(device) for k, v in batch.inputs.items()}
        targets = batch.targets.to(device) if batch.targets is not None else None
        meta = self._move_to_device(batch.meta, device)

        return Batch(inputs=inputs, targets=targets, meta=meta)
    
    def _move_to_device(self, obj: Any, device: torch.device) -> Any:
        """Recursively move all torch.Tensors inside obj to device."""
        if torch.is_tensor(obj):
            return obj.to(device)

        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            t = [self._move_to_device(v, device) for v in obj]
            return type(obj)(t)  # preserves list/tuple

        # leave everything else as-is (str, int, float, None, Path, etc.)
        return obj

    @staticmethod
    def _sanitize_split_name(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("._-")
        return name or "split"

    @classmethod
    def _normalize_loaders(
        cls,
        loaders: Optional[Union[DataLoader, Mapping[str, DataLoader], List[DataLoader], Tuple[DataLoader, ...]]],
        *,
        default_name: str,
    ) -> Dict[str, DataLoader]:
        if loaders is None:
            return {}
        if isinstance(loaders, DataLoader):
            return {default_name: loaders}
        if isinstance(loaders, dict):
            out = {}
            for k, v in loaders.items():
                key = cls._sanitize_split_name(str(k))
                if key in out:
                    i = 2
                    new_key = f"{key}_{i}"
                    while new_key in out:
                        i += 1
                        new_key = f"{key}_{i}"
                    key = new_key
                out[key] = v
            return out
        if isinstance(loaders, (list, tuple)):
            return {f"{default_name}{i}": dl for i, dl in enumerate(loaders)}
        raise TypeError(f"Unsupported loaders type: {type(loaders)}")

    @staticmethod
    def _scheduler_needs_metric(scheduler: Any) -> bool:
        """
        True if scheduler.step() expects a metric (e.g., ReduceLROnPlateau).
        Works generically via signature inspection.
        """
        try:
            sig = inspect.signature(scheduler.step)
            params = list(sig.parameters.values())
            # params[0] is usually 'self'
            # ReduceLROnPlateau: step(metrics, epoch=None) -> requires metrics
            if len(params) >= 2:
                p1 = params[0]
                # required positional or named 'metrics'
                if p1.default is inspect._empty or p1.name == "metrics":
                    return True
            return False
        except Exception:
            return False

    def _scheduler_step(self, logs: Optional[Mapping[str, Any]]) -> None:
        if self.scheduler is None or not self.config.use_scheduler:
            return

        needs_metric = self._scheduler_needs_metric(self.scheduler)
        monitor = getattr(self.config, "scheduler_monitor", None)

        if needs_metric:
            if logs is None or not monitor or monitor not in logs:
                return
            metric = logs[monitor]
            try:
                metric = float(metric)
            except Exception:
                pass
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
