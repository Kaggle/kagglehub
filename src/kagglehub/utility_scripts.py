import logging
import sys
from http import HTTPStatus

from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

from kagglehub import registry
from kagglehub.clients import build_kaggle_client
from kagglehub.exceptions import KaggleApiHTTPError, handle_call
from kagglehub.handle import UtilityScriptHandle, parse_utility_script_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def utility_script_install(handle: str, *, force_download: bool | None = False) -> str:
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
        with build_kaggle_client() as api_client:
            r = ApiGetKernelRequest()
            r.user_name = h.owner
            r.kernel_slug = h.notebook
            response = handle_call(lambda: api_client.kernels.kernels_api_client.get_kernel(r))
            return "utility script" in response.metadata.category_ids

    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not find '{h.owner}' metadata by user '{h.notebook}'.")
            return False
        else:
            raise (e)
