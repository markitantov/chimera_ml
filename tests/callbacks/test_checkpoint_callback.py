from pathlib import Path

import torch

from chimera_ml.callbacks.checkpoint_callback import CheckpointCallback


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
    def __init__(self, with_scheduler: bool = False, with_logger: bool = True):
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1) if with_scheduler else None
        self.global_step = 7
        self.logger = _LoggerStub() if with_logger else None


def test_checkpoint_saves_last_and_topk_best(tmp_path: Path):
    trainer = _TrainerStub()
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    cb.on_fit_start(trainer)

    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 1.0})
    cb.on_epoch_end(trainer, epoch=2, logs={"val/loss": 0.8})

    ckpt_dir = tmp_path / "exp" / "run" / "checkpoints"
    files = sorted(p.name for p in ckpt_dir.glob("*.pt"))

    assert "last.pt" in files
    best_files = [f for f in files if f != "last.pt"]
    assert len(best_files) == 1
    assert cb._best == 0.8
    assert len(cb._saved) == 1


def test_checkpoint_missing_monitor_logs_warning_and_still_saves_last(tmp_path: Path):
    trainer = _TrainerStub()
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    cb.on_fit_start(trainer)

    cb.on_epoch_end(trainer, epoch=1, logs={"train/loss": 0.5})

    ckpt_dir = tmp_path / "exp" / "run" / "checkpoints"
    files = sorted(p.name for p in ckpt_dir.glob("*.pt"))
    assert files == ["last.pt"]
    assert any("monitor='val/loss' not found" in msg for msg in trainer.logger.warnings)


def test_checkpoint_payload_contains_scheduler_state(tmp_path: Path):
    trainer = _TrainerStub(with_scheduler=True)
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    cb.on_fit_start(trainer)
    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 1.0})

    best_path = cb._saved[0]
    payload = torch.load(best_path, map_location="cpu")
    assert "scheduler_state_dict" in payload
    assert payload["global_step"] == trainer.global_step


def test_checkpoint_missing_monitor_with_no_logger_does_not_crash(tmp_path: Path):
    trainer = _TrainerStub(with_logger=False)
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    cb.on_fit_start(trainer)

    cb.on_epoch_end(trainer, epoch=1, logs={"train/loss": 0.5})

    ckpt_dir = tmp_path / "exp" / "run" / "checkpoints"
    files = sorted(p.name for p in ckpt_dir.glob("*.pt"))
    assert files == ["last.pt"]


def test_checkpoint_mode_max_tracks_larger_metric(tmp_path: Path):
    trainer = _TrainerStub()
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    cb.on_fit_start(trainer)

    cb.on_epoch_end(trainer, epoch=1, logs={"val/acc": 0.70})
    cb.on_epoch_end(trainer, epoch=2, logs={"val/acc": 0.90})
    cb.on_epoch_end(trainer, epoch=3, logs={"val/acc": 0.80})

    assert cb._best == 0.90
    assert len(cb._saved) == 1


def test_checkpoint_save_top_k_zero_keeps_all_best_improvements(tmp_path: Path):
    trainer = _TrainerStub()
    cb = CheckpointCallback(
        log_path=str(tmp_path),
        experiment_name="exp",
        run_name="run",
        monitor="val/loss",
        mode="min",
        save_top_k=0,
        save_last=False,
    )
    cb.on_fit_start(trainer)

    cb.on_epoch_end(trainer, epoch=1, logs={"val/loss": 1.0})
    cb.on_epoch_end(trainer, epoch=2, logs={"val/loss": 0.9})
    cb.on_epoch_end(trainer, epoch=3, logs={"val/loss": 0.8})

    assert cb._best == 0.8
    assert len(cb._saved) == 3
