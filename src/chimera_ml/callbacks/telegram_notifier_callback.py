import html
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from chimera_ml.callbacks.base import BaseCallback
from chimera_ml.core.registry import CALLBACKS


def _env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _format_metric(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


@dataclass
class TelegramNotifierCallback(BaseCallback):
    """
    Sends a Telegram message when training finishes (on_fit_end).
    Tracks best epoch using the same monitor logic as EarlyStoppingCallback.

    Notes:
    - Grabs MLflow experiment name if available (trainer.mlflow_logger.experiment_name).
    - Optionally includes trainer progress info (epoch/global_step) if present.
    - Optionally includes the last available metrics dict from on_epoch_end.
    """

    token_env_field: str = "TELEGRAM_BOT_TOKEN"
    chat_id_env_field: str = "TELEGRAM_CHAT_ID"

    message_prefix: str = "✅ Training finished"
    message_suffix: str = ""

    include_mlflow_experiment_name: bool = True
    include_trainer_progress: bool = True
    include_last_logs: bool = True
    include_best_logs: bool = True

    monitor: str = "val/loss"
    mode: str = "min"  # "min" or "max"
    min_delta: float = 0.0

    request_timeout_sec: int = 15
    telegram_parse_mode: str = "HTML"

    _exp_name: str | None = field(init=False, default=None)
    _last_logs: dict[str, float] = field(init=False, default_factory=dict)

    _best_epoch: Any | None = field(init=False, default=None)
    _best_value: float | None = field(init=False, default=None)
    _best_logs: dict[str, float] = field(init=False, default_factory=dict)
    _session: requests.Session | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

    def on_fit_start(self, trainer: Any) -> None:
        """Initialize Telegram session and reset run-level state."""
        self._telegram_token = _env_required(self.token_env_field)
        self._telegram_chat_id = _env_required(self.chat_id_env_field)

        # Prepare Telegram API endpoint and HTTP session (reused connection).
        self._api_url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        self._session = requests.Session()

        # Cache MLflow experiment name if present.
        self._exp_name = None
        mlflow_logger = getattr(trainer, "mlflow_logger", None)
        if self.include_mlflow_experiment_name and mlflow_logger is not None:
            self._exp_name = getattr(mlflow_logger, "experiment_name", None)

        # Keep the last epoch logs (if provided by the trainer).
        self._last_logs = {}
        self._best_epoch = None
        self._best_value = None
        self._best_logs = {}

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)

    def on_epoch_end(self, trainer: Any, epoch: int, logs: dict[str, float]) -> None:
        """Track last logs and best monitor value during training."""
        if not logs:
            return

        self._last_logs = dict(logs)

        if self.monitor not in logs:
            available = ", ".join(sorted(logs.keys()))
            self._warning(
                trainer,
                f"[TelegramNotifierCallback] monitor='{self.monitor}' not found in logs. "
                f"Available keys: {available}",
            )

            return

        current = float(logs[self.monitor])

        if self._best_value is None:
            self._best_value = current
            self._best_epoch = epoch
            self._best_logs = dict(logs)
            return

        if self._is_improvement(current, self._best_value):
            self._best_value = current
            self._best_epoch = epoch
            self._best_logs = dict(logs)

    def _build_message(self, trainer: Any) -> str:
        lines: list[str] = [f"<b>{html.escape(self.message_prefix)}</b>"]

        # Add experiment name if available.
        if self._exp_name:
            lines.append(f"<b>Experiment:</b> <code>{html.escape(str(self._exp_name))}</code>")

        # Add progress info if trainer exposes it.
        if self.include_trainer_progress:
            last_epoch = getattr(trainer, "current_epoch", None)
            if last_epoch is None:
                last_epoch = getattr(trainer, "epoch", None)

            global_step = getattr(trainer, "global_step", None)

            if last_epoch is not None:
                lines.append(
                    f"<b>Last epoch:</b> <code>{html.escape(_format_scalar(last_epoch))}</code>"
                )
            if global_step is not None:
                lines.append(
                    f"<b>Global step:</b> <code>{html.escape(_format_scalar(global_step))}</code>"
                )

        if self._best_value is not None:
            lines.append(
                f"<b>Best epoch:</b> <code>{html.escape(_format_scalar(self._best_epoch))}</code>"
            )
            lines.append(f"<b>Monitor:</b> <code>{html.escape(self.monitor)}</code>")
            lines.append(f"<b>Mode:</b> <code>{html.escape(self.mode)}</code>")
            lines.append(
                f"<b>Best value:</b> <code>{html.escape(_format_scalar(self._best_value))}</code>"
            )

            if self.include_best_logs and self._best_logs:
                lines.append("<b>Metrics at best epoch:</b>")
                for key in sorted(self._best_logs):
                    value = self._best_logs[key]
                    lines.append(
                        f"• <code>{html.escape(str(key))}</code>: "
                        f"<code>{html.escape(_format_metric(value))}</code>"
                    )

        # Add last logs/metrics if we have them.
        if self.include_last_logs and self._last_logs:
            lines.append("<b>Last metrics:</b>")
            for key in sorted(self._last_logs):
                value = self._last_logs[key]
                lines.append(
                    f"• <code>{html.escape(str(key))}</code>: "
                    f"<code>{html.escape(_format_metric(value))}</code>"
                )

        if self.message_suffix:
            lines.append("")
            lines.append(html.escape(self.message_suffix))

        return "\n".join(lines)

    def on_fit_end(self, trainer: Any) -> None:
        """Build and send the final Telegram message."""
        if self._session is None:
            self._warning(
                trainer, "[TelegramNotifierCallback] Session is not initialized; skipping send."
            )
            return

        text = self._build_message(trainer)
        data = {
            "chat_id": self._telegram_chat_id,
            "text": text,
            "parse_mode": self.telegram_parse_mode,
        }

        resp = None
        try:
            resp = self._session.post(self._api_url, data=data, timeout=self.request_timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as e:
            status = getattr(resp, "status_code", None) if "resp" in locals() else None
            body = None
            try:
                body = resp.text if "resp" in locals() else None
            except Exception:
                body = None

            self._warning(trainer, f"[TelegramNotifierCallback] Failed to send message: {e}")
            if status is not None:
                self._info(trainer, f"[TelegramNotifierCallback] HTTP status: {status}")
            if body:
                self._info(trainer, f"[TelegramNotifierCallback] Response body: {body}")
        finally:
            self._session.close()
            self._session = None


@CALLBACKS.register("telegram_notifier_callback")
def telegram_notifier_callback(**params):
    return TelegramNotifierCallback(**params)
