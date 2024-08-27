import logging
from typing import Optional, Union

from kagglehub import registry
from kagglehub.gcs_upload import normalize_patterns, upload_files_and_directories
from kagglehub.handle import parse_model_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK
from kagglehub.models_helpers import create_model_if_missing, create_model_instance_or_version

logger = logging.getLogger(__name__)

# Patterns that are always ignored for model uploading.
DEFAULT_IGNORE_PATTERNS = [".git/", "*/.git/", ".cache/", ".huggingface/"]


def model_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """Download model files.

    Args:
        handle: (string) the model handle.
        path: (string) Optional path to a file within the model bundle.
        force_download: (bool) Optional flag to force download a model, even if it's cached.


    Returns:
        A string representing the path to the requested model files.
    """
    h = parse_model_handle(handle)
    logger.info(f"Downloading Model: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    return registry.model_resolver(h, path, force_download=force_download)


def model_upload(
    handle: str,
    local_model_dir: str,
    license_name: Optional[str] = None,
    version_notes: str = "",
    ignore_patterns: Optional[Union[list[str], str]] = None,
) -> None:
    """Upload model files.

    Args:
        handle: (str) the model handle.
        local_model_dir: (str) path to a file in a local directory.
        license_name: (str) model license.
        version_notes: (str, optional) model versions.
        ignore_patterns (str or list[str], optional):
            Additional ignore patterns to DEFAULT_IGNORE_PATTERNS.
            Files matching any of the patterns are not uploaded.
            Patterns are standard wildcards that can be matched by
            https://docs.python.org/3/library/fnmatch.html.
            Use a pattern ending with "/" to ignore the whole dir,
            e.g., ".git/" is equivalent to ".git/*".
    """
    # parse slug
    h = parse_model_handle(handle)
    logger.info(f"Uploading Model {h.to_url()} ...")
    if h.is_versioned():
        is_versioned_exception = "The model handle should not include the version"
        raise ValueError(is_versioned_exception)

    # Create the model if it doesn't already exist
    create_model_if_missing(h.owner, h.model)

    # Upload the model files to GCS

    tokens = upload_files_and_directories(
        local_model_dir,
        item_type="model",
        ignore_patterns=normalize_patterns(default=DEFAULT_IGNORE_PATTERNS, additional=ignore_patterns),
    )

    # Create a model instance if it doesn't exist, and create a new instance version if an instance exists
    create_model_instance_or_version(h, tokens, license_name, version_notes)
