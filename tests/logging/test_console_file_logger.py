import logging
from pathlib import Path

from chimera_ml.core.registry import LOGGERS
from chimera_ml.logging.console_file_logger import ConsoleFileLogger, _level, console_file_logger


def test_level_parses_int_and_string():
    assert _level(logging.WARNING) == logging.WARNING
    assert _level("debug") == logging.DEBUG
    assert _level("unknown") == logging.INFO


def test_console_file_logger_creates_file_and_writes_messages(tmp_path: Path):
    logger = ConsoleFileLogger(
        log_path=tmp_path,
        experiment_name="exp",
        run_name="run",
        log_file="train.log",
        name="chimera_test_logger",
        file_mode="w",
        console_level="ERROR",
        file_level="INFO",
    )

    logger.info("hello")
    logger.warning("warn")

    path = tmp_path / "exp" / "run" / "train.log"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "hello" in text
    assert "warn" in text


def test_console_file_logger_replaces_old_handlers(tmp_path: Path):
    first = ConsoleFileLogger(log_path=tmp_path, name="same_logger", file_mode="w")
    first.info("first")

    second = ConsoleFileLogger(log_path=tmp_path, name="same_logger", file_mode="w")
    second.info("second")

    # Exactly file + stream handlers after re-init, no duplicates.
    assert len(second.logger.handlers) == 2
    assert second.logger.propagate is False


def test_console_file_logger_getattr_and_factory_registry(tmp_path: Path):
    logger = console_file_logger(log_path=str(tmp_path), name="factory_logger")
    logger.setLevel(logging.ERROR)
    assert logger.level == logging.ERROR

    factory = LOGGERS.get("console_file_logger")
    assert callable(factory)
    built = factory(log_path=str(tmp_path), name="from_registry")
    assert isinstance(built, ConsoleFileLogger)
