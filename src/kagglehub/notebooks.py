import logging

from kagglehub import registry
from kagglehub.handle import parse_notebook_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def notebook_output_download(
    handle: str,
    path: str | None = None,
    *,
    force_download: bool | None = False,
    output_dir: str | None = None,
) -> str:
    """Download notebook output files.

    Args:
        handle: (string) the notebook handle under https://kaggle.com/code.
        path: (string) Optional path to a file within the notebook output.
        force_download: (bool) Optional flag to force download a notebook output, even if it's cached or already in
            output_dir.
        output_dir: (string) Optional output directory for direct download, bypassing the default cache.


    Returns:
        A string representing the path to the requested notebook output files.
    """
    h = parse_notebook_handle(handle)
    logger.info(f"Downloading Notebook Output: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    path, _ = registry.notebook_output_resolver(
        h,
        path,
        force_download=force_download,
        output_dir=output_dir,
    )
    return path
