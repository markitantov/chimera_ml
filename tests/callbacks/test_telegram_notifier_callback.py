import importlib

import pytest
import requests

from chimera_ml.callbacks.telegram_notifier_callback import TelegramNotifierCallback

telegram_module = importlib.import_module("chimera_ml.callbacks.telegram_notifier_callback")


class _LoggerStub:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.infos.append(str(msg))

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.warnings.append(str(msg))


class _MlflowStub:
    experiment_name = "exp_name"


class _TrainerStub:
    def __init__(self, with_logger: bool = True, with_mlflow: bool = True):
        self.logger = _LoggerStub() if with_logger else None
        self.mlflow_logger = _MlflowStub() if with_mlflow else None
        self.current_epoch = 3
        self.global_step = 42


class _ResponseStub:
    def __init__(self, ok: bool):
        self.status_code = 500 if not ok else 200
        self.text = "boom" if not ok else "ok"
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("bad status")


class _SessionStub:
    def __init__(self, ok: bool = True):
        self.ok = ok
        self.closed = False
        self.calls = []

    def post(self, url, data, timeout):
        self.calls.append((url, data, timeout))
        return _ResponseStub(ok=self.ok)

    def close(self):
        self.closed = True


def test_telegram_on_fit_start_requires_env_vars(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    cb = TelegramNotifierCallback()

    with pytest.raises(RuntimeError, match="Missing required env var"):
        cb.on_fit_start(_TrainerStub())


def test_telegram_callback_no_logger_and_failed_request_does_not_crash(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    session = _SessionStub(ok=False)
    monkeypatch.setattr(telegram_module.requests, "Session", lambda: session)

    trainer = _TrainerStub(with_logger=False)
    cb = TelegramNotifierCallback()
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 0.5})
    cb.on_fit_end(trainer)

    assert len(session.calls) == 1
    assert session.closed is True


def test_telegram_callback_uses_mlflow_logger_experiment_name(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    session = _SessionStub(ok=True)
    monkeypatch.setattr(telegram_module.requests, "Session", lambda: session)

    trainer = _TrainerStub(with_logger=True, with_mlflow=True)
    cb = TelegramNotifierCallback()
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 0.25, "train/loss": 0.5})
    cb.on_fit_end(trainer)

    sent_data = session.calls[0][1]
    assert "Experiment:" in sent_data["text"]
    assert "exp_name" in sent_data["text"]
    assert session.closed is True


def test_telegram_callback_without_mlflow_does_not_render_experiment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    session = _SessionStub(ok=True)
    monkeypatch.setattr(telegram_module.requests, "Session", lambda: session)

    trainer = _TrainerStub(with_logger=True, with_mlflow=False)
    cb = TelegramNotifierCallback()
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 0.25, "train/loss": 0.5})
    cb.on_fit_end(trainer)

    sent_data = session.calls[0][1]
    assert "Experiment:" not in sent_data["text"]
    assert session.closed is True


def test_telegram_monitor_missing_logs_warning(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    session = _SessionStub(ok=True)
    monkeypatch.setattr(telegram_module.requests, "Session", lambda: session)

    trainer = _TrainerStub(with_logger=True, with_mlflow=True)
    cb = TelegramNotifierCallback(monitor="val/loss")
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, epoch=1, logs={"train/loss": 0.5})

    assert any("monitor='val/loss' not found" in msg for msg in trainer.logger.warnings)


def test_telegram_on_fit_end_without_on_fit_start_does_not_crash():
    trainer = _TrainerStub(with_logger=True, with_mlflow=False)
    cb = TelegramNotifierCallback()

    cb.on_fit_end(trainer)
    assert any("Session is not initialized" in msg for msg in trainer.logger.warnings)
