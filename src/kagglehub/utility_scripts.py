import logging
import sys
from typing import Optional

from kagglehub import registry
from kagglehub.handle import parse_utility_script_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def utility_script_install(handle: str, *, force_download: Optional[bool] = False) -> str:
    """
    Downloads the utility script and adds the folder path to the system path

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        force_download: (bool) Optional flag to force download motebook output, even if it's cached.


    Returns:
        A string representing the path to the requested notebook output files.
    """
    h = parse_utility_script_handle(handle)

    logger.info(f"Downloading Utility Script: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    utility_script_path = registry.notebook_output_resolver(h, path=None, force_download=force_download)

    sys.path.append(utility_script_path)

    return utility_script_path
