import logging

from kagglehub.config import get_log_verbosity


def _configure_logger():
    library_name = __name__.split(".")[0]  # i.e. "kagglehub"
    library_logger = logging.getLogger(library_name)
    library_logger.addHandler(logging.StreamHandler())
    library_logger.setLevel(get_log_verbosity())


_configure_logger()
