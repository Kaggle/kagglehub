import logging

from kagglehub.config import get_log_verbosity


def _configure_logger() -> None:
    library_name = __name__.split(".")[0]  # i.e. "kagglehub"
    library_logger = logging.getLogger(library_name)
    library_logger.addHandler(logging.StreamHandler())
    # Disable propagation of the library log outputs.
    # This prevents the same message again from being printed again if a root logger is defined.
    library_logger.propagate = False
    library_logger.setLevel(get_log_verbosity())


_configure_logger()
