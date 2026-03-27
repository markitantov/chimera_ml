from chimera_ml.callbacks.early_stopping_callback import EarlyStoppingCallback


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


class _TrainerStub:
    def __init__(self, with_logger: bool = True):
        self.logger = _LoggerStub() if with_logger else None
        self.stop_training = False


def test_early_stopping_stops_after_patience():
    cb = EarlyStoppingCallback(monitor="val/loss", mode="min", patience=2, min_delta=0.0)
    trainer = _TrainerStub()

    cb.on_epoch_end(trainer, 1, {"val/loss": 1.0})
    cb.on_epoch_end(trainer, 2, {"val/loss": 1.1})
    cb.on_epoch_end(trainer, 3, {"val/loss": 1.2})

    assert cb._should_stop is True
    assert trainer.stop_training is True
    assert cb._bad_epochs == 2
    assert cb._countdown == 0


def test_early_stopping_improvement_resets_bad_epochs():
    cb = EarlyStoppingCallback(monitor="val/loss", mode="min", patience=3, min_delta=0.0)
    trainer = _TrainerStub()

    cb.on_epoch_end(trainer, 1, {"val/loss": 1.0})
    cb.on_epoch_end(trainer, 2, {"val/loss": 1.2})
    assert cb._bad_epochs == 1

    cb.on_epoch_end(trainer, 3, {"val/loss": 0.9})
    assert cb._bad_epochs == 0
    assert cb._best == 0.9
    assert cb._should_stop is False


def test_early_stopping_monitor_missing_warns():
    cb = EarlyStoppingCallback(monitor="val/loss", mode="min", patience=2)
    trainer = _TrainerStub()

    cb.on_epoch_end(trainer, 1, {"train/loss": 0.5})
    assert trainer.stop_training is False
    assert cb._should_stop is False
    assert any("monitor='val/loss' not found" in msg for msg in trainer.logger.warnings)


def test_early_stopping_works_without_logger():
    cb = EarlyStoppingCallback(monitor="val/loss", mode="min", patience=2, min_delta=0.0)
    trainer = _TrainerStub(with_logger=False)

    cb.on_epoch_end(trainer, 1, {"val/loss": 1.0})
    cb.on_epoch_end(trainer, 2, {"val/loss": 1.1})
    cb.on_epoch_end(trainer, 3, {"val/loss": 1.2})

    assert cb._should_stop is True
    assert trainer.stop_training is True


def test_early_stopping_mode_max_stops_when_metric_decreases():
    cb = EarlyStoppingCallback(monitor="val/acc", mode="max", patience=2, min_delta=0.0)
    trainer = _TrainerStub()

    cb.on_epoch_end(trainer, 1, {"val/acc": 0.8})
    cb.on_epoch_end(trainer, 2, {"val/acc": 0.7})
    cb.on_epoch_end(trainer, 3, {"val/acc": 0.6})

    assert cb._should_stop is True
    assert trainer.stop_training is True
    assert cb._best == 0.8


def test_early_stopping_min_delta_threshold_is_respected():
    cb = EarlyStoppingCallback(monitor="val/loss", mode="min", patience=2, min_delta=0.1)
    trainer = _TrainerStub()

    cb.on_epoch_end(trainer, 1, {"val/loss": 1.0})
    cb.on_epoch_end(trainer, 2, {"val/loss": 0.95})  # not enough improvement (needs < 0.9)
    cb.on_epoch_end(trainer, 3, {"val/loss": 0.96})

    assert cb._best == 1.0
    assert cb._bad_epochs == 2
    assert cb._should_stop is True
