import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Any

from chimera_ml.core.registry import LOGGERS


def _level(x: Union[int, str]) -> int:
    if isinstance(x, int):
        return x
    return logging._nameToLevel.get(str(x).upper(), logging.INFO)


@dataclass
class ConsoleFileLogger:
    """Create console + file logger with timestamps.
    - Creates parent dirs for log_path
    - Clears previous handlers (no duplicates)
    - Disables propagation (avoids double printing)
    """
    log_path: str | Path
    experiment_name: str = 'chimera'
    run_name: str = 'train'
    log_file: str = 'train.log'
    name: str = "chimera"
    format: str = "%(asctime)s:%(levelname)s:%(message)s"
    console_level: Union[int, str] = logging.INFO
    file_level: Union[int, str] = logging.INFO
    file_mode: str = "a"
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path) / self.experiment_name / self.run_name / Path(self.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(self.format)

        file_handler = logging.FileHandler(self.log_path, mode=self.file_mode, encoding=self.encoding)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(_level(self.file_level))

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(_level(self.console_level))

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # remove old handlers to avoid duplicates
        for h in list(logger.handlers):
            logger.removeHandler(h)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger = logger

    # --- proxy methods ---
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

    def __getattr__(self, name: str):
        # delegate any other logger methods (critical, setLevel, handlers, etc.)
        return getattr(self.logger, name)


@LOGGERS.register("console_file_logger")
def console_file_logger(
    *,
    log_path: str,
    name: str = "chimera",
    experiment_name: str = 'chimera',
    run_name: str = 'train',
    log_file: str = 'train.log',
    format: str = "%(asctime)s:%(levelname)s:%(message)s",
    console_level: Union[int, str] = logging.INFO,
    file_level: Union[int, str] = logging.INFO,
    file_mode: str = "a",
    encoding: str = "utf-8",
    **_,
) -> ConsoleFileLogger:
    return ConsoleFileLogger(
        log_path=Path(log_path),
        name=name,
        experiment_name=experiment_name,
        run_name=run_name,
        log_file=log_file,
        format=format,
        console_level=console_level,
        file_level=file_level,
        file_mode=file_mode,
        encoding=encoding,
    )

