import logging
from typing import Optional

from kagglehub import registry
from kagglehub.handle import parse_notebook_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def notebook_output_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """[WORK IN PROGRESS]

    Download notebook output files.

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        path: (string) Optional path to a file within the notebook output.
        force_download: (bool) Optional flag to force download motebook output, even if it's cached.


    Returns:
        A string representing the path to the requested notebook output files.
    """
    h = parse_notebook_handle(handle)
    logger.info(f"Downloading Notebook Output: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    return registry.notebook_output_resolver(h, path, force_download=force_download)
