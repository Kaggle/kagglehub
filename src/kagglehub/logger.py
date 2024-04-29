import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from kagglehub.config import get_log_verbosity


def _configure_logger() -> None:
    library_name = __name__.split(".")[0]  # i.e. "kagglehub"
    library_logger = logging.getLogger(library_name)
    while library_logger.handlers:
        library_logger.handlers.pop()

    log_dir = Path.home() / ".kaggle" / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    file_handler = RotatingFileHandler(str(log_dir / "kagglehub.log"), maxBytes=1024 * 1024 * 5, backupCount=5)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    library_logger.addHandler(file_handler)
    library_logger.addHandler(logging.StreamHandler())
    # Disable propagation of the library log outputs.
    # This prevents the same message again from being printed again if a root logger is defined.
    library_logger.propagate = False
    library_logger.setLevel(get_log_verbosity())


_configure_logger()
