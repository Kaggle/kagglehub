import logging
from typing import Optional

from kagglehub import registry
from kagglehub.handle import parse_dataset_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def competition_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """Download competition files
    Args:
        handle: (string) the competition name
        path: (string) Optional path to a file within a competition
        force_download: (bool) Optional flag to force download a competition, even if it's cached
    Returns:
        A string requesting the path to the requested competition files.
    """

    h = parse_dataset_handle(handle)
    logger.info(f"Downloading competition: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    return registry.dataset_resolver(h, path, force_download=force_download)
