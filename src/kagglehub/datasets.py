import logging
from typing import Optional

from kagglehub import registry
<<<<<<< HEAD
from kagglehub.datasets_helpers import create_dataset_or_version
from kagglehub.gcs_upload import upload_files_and_directories
=======
from kagglehub.datasets_helpers import create_dataset_if_missing, create_dataset_instance_or_version
from kagglehub.gcs_upload import upload_files
>>>>>>> 5f73ba0 (Lint fix)
from kagglehub.handle import parse_dataset_handle

logger = logging.getLogger(__name__)


def dataset_download(handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False) -> str:
    """Download dataset files
    Args:
        handle: (string) the dataset handle
        path: (string) Optional path to a file within the dataset
        force_download: (bool) Optional flag to force download a dataset, even if it's cached.
    Returns:
        A string requesting the path to the requested dataset files.
    """

    h = parse_dataset_handle(handle)
    return registry.dataset_resolver(h, path, force_download=force_download)


def dataset_upload(handle: str, local_dataset_dir: str, version_notes: str = "") -> None:
    """Upload dataset files.
    Args:
        handle: (string) the dataset handle.
        local_dataset_dir: (string) path to a file in a local directory.
        version_notes: (string) Optional to write dataset versions.
    """
    h = parse_dataset_handle(handle)
<<<<<<< HEAD
    logger.info(f"Uploading Dataset {h.to_url()} ...")
    if h.is_versioned():
        is_versioned_exception = "The dataset handle should not include the version"
        raise ValueError(is_versioned_exception)
=======

    # Create the dataset if it doesn't already exist
    create_dataset_if_missing(h.owner, h.dataset_name)
>>>>>>> 5f73ba0 (Lint fix)

    # Upload the dataset files to GCS
    tokens = upload_files_and_directories(local_dataset_dir, "dataset")

    create_dataset_or_version(h, tokens, version_notes)
