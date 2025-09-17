import logging

from kagglehub import registry
from kagglehub.handle import parse_competition_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def competition_download(handle: str, path: str | None = None, *, force_download: bool | None = False) -> str:
    """Download competition dataset
    Args:
        handle: (string) the competition name
        path: (string) Optional path to a file within a competition dataset
        force_download: (bool) Optional flag to force download a competition dataset, even if it's cached
    Returns:
        A string requesting the path to the requested competition files.
    """

    h = parse_competition_handle(handle)
    logger.info(f"Downloading competition: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    path, _ = registry.competition_resolver(h, path, force_download=force_download)
    return path
