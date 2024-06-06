import logging
import sys
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, List, Optional

from kagglehub.config import get_log_verbosity

_FILE_BLOCK_KEY = "kaggle_file"
EXTRA_FILE_BLOCK = {"block": _FILE_BLOCK_KEY}
_CONSOLE_BLOCK_KEY = "console"
EXTRA_CONSOLE_BLOCK = {"block": _CONSOLE_BLOCK_KEY}


def _block_logrecord_factory(elements: List[str]) -> Callable[[LogRecord], bool]:
    """Filter to block log statements based on data attributes
    Args:
        elements: The value for the key 'block'. For example log.info("..", extra={"block" : "console"})
    """

    def _filter(record: LogRecord) -> bool:
        if hasattr(record, "block"):
            if record.block in elements:
                return False
        return True

    return _filter


def _configure_logger(log_dir: Optional[Path] = None) -> None:
    library_name = __name__.split(".")[0]  # i.e. "kagglehub"
    library_logger = logging.getLogger(library_name)
    while library_logger.handlers:
        handler = library_logger.handlers.pop()
        while handler.filters:
            handler.filters.pop()
    log_dir = Path.home() / ".kaggle" / "logs" if log_dir is None else log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    file_handler = RotatingFileHandler(
        str(log_dir / "kagglehub.log"), maxBytes=1024 * 1024 * 5, backupCount=5, delay=True
    )
    file_handler.addFilter(_block_logrecord_factory([_FILE_BLOCK_KEY]))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    library_logger.addHandler(file_handler)

    sh = logging.StreamHandler(sys.stdout)
    sh.addFilter(_block_logrecord_factory([_CONSOLE_BLOCK_KEY]))
    sh.setLevel(get_log_verbosity())
    library_logger.addHandler(sh)
    # Disable propagation of the library log outputs.
    # This prevents the same message again from being printed again if a root logger is defined.
    library_logger.propagate = False
    library_logger.setLevel(get_log_verbosity())


_configure_logger()
