import logging
from typing import Optional

from kagglehub import registry
from kagglehub.gcs_upload import upload_files_and_directories
from kagglehub.handle import parse_model_handle
from kagglehub.models_helpers import create_model_if_missing, create_model_instance_or_version

logger = logging.getLogger(__name__)


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
    return registry.model_resolver(h, path, force_download=force_download)


def model_upload(
    handle: str, local_model_dir: str, license_name: Optional[str] = None, version_notes: str = ""
) -> None:
    """Upload model files.

    Args:
        handle: (string) the model handle.
        local_model_dir: (string) path to a file in a local directory.
        license_name: (string) model license.
        version_notes: (string) Optional to write to model versions.
    """
    # parse slug
    h = parse_model_handle(handle)

    if h.is_versioned():
        is_versioned_exception = "The model handle should not include the version"
        raise ValueError(is_versioned_exception)

    # Create the model if it doesn't already exist
    create_model_if_missing(h.owner, h.model)

    # Upload the model files to GCS
    tokens = upload_files_and_directories(local_model_dir, "model")

    # Create a model instance if it doesn't exist, and create a new instance version if an instance exists
    create_model_instance_or_version(h, tokens, license_name, version_notes)
