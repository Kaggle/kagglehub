import logging
import sys
from http import HTTPStatus
from typing import Optional

from kagglehub import registry
from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import UtilityScriptHandle, parse_utility_script_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def utility_script_install(handle: str, *, force_download: Optional[bool] = False) -> str:
    """
    Downloads the utility script and adds the directory path to the system path.

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        force_download: (bool) Optional flag to force download motebook output, even if it's cached.


    Returns:
        A string representing the path to the requested notebook output files.
    """
    h = parse_utility_script_handle(handle)

    logger.info(f"Downloading Utility Script: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    utility_script_path, _ = registry.notebook_output_resolver(h, path=None, force_download=force_download)

    if not _is_notebook_utility_script(h):
        logger.info(
            f"Notebook '{h.notebook}' by user '{h.owner}' is not a Utility Script"
            "\n and will not be added to system path"
        )
        return utility_script_path

    if utility_script_path not in sys.path:
        sys.path.append(utility_script_path)

    logger.info(f"Added {utility_script_path} to system path")
    return utility_script_path


def _is_notebook_utility_script(h: UtilityScriptHandle) -> bool:
    try:
        api_client = KaggleApiV1Client()
        json_response = api_client.get(f"kernels/pull/{h.owner}/{h.notebook}", h)

        category_ids = json_response["metadata"]["categoryIds"]
        return "utility script" in category_ids

    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not find '{h.owner}' metadata by user '{h.notebook}'.")
            return False
        else:
            raise (e)
