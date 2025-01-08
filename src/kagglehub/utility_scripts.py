import logging
import sys
from http import HTTPStatus
from typing import Optional

from kagglehub import registry
from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import parse_utility_script_handle
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
    utility_script_path = registry.notebook_output_resolver(h, path=None, force_download=force_download)

    if _is_notebook_utility_script(h.owner, h.notebook):
        sys.path.append(utility_script_path)
    else:
        logger.info("Notebook is not an Utility Script")

    return utility_script_path


def _is_notebook_utility_script(user_name: str, notebook_slug: str) -> bool:
    try:
        api_client = KaggleApiV1Client()

        json_response = api_client.get(f"kernels/pull/{user_name}/{notebook_slug}")

        category_ids = json_response["metadata"]["categoryIds"]
        if "utility script" in category_ids:
            return True
        else:
            return False

    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not find '{notebook_slug}' metadata for user '{user_name}'.")
            return False
        else:
            raise (e)
