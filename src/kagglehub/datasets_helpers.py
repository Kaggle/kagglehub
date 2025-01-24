import logging
from http import HTTPStatus
from typing import Optional

from kagglehub import registry
from kagglehub.clients import BackendError, KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.gcs_upload import UploadDirectoryInfo
from kagglehub.handle import DatasetHandle, parse_dataset_handle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


def _create_dataset(dataset_handle: DatasetHandle, files_and_directories: UploadDirectoryInfo) -> None:
    data = {
        "ownerSlug": dataset_handle.owner,
        "title": dataset_handle.dataset,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
        "isPrivate": True,
    }

    api_client = KaggleApiV1Client()
    api_client.post("/datasets/create/new", data)
    logger.info(
        f"Your dataset instance has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}"
    )


def _create_dataset_version(
    dataset_handle: DatasetHandle, files_and_directories: UploadDirectoryInfo, version_notes: str = ""
) -> None:
    data = {
        "versionNotes": version_notes,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
    }
    api_client = KaggleApiV1Client()
    api_client.post(f"/datasets/create/version/{dataset_handle.owner}/{dataset_handle.dataset}", data)
    logger.info(f"Your dataset has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}")


def create_dataset_or_version(
    dataset_handle: DatasetHandle, files: UploadDirectoryInfo, version_notes: str = ""
) -> None:
    try:
        _create_dataset(dataset_handle, files)
    except BackendError as e:
        if e.error_code in (None, HTTPStatus.CONFLICT):
            # Dataset already exists, creating a new version instead.
            _create_dataset_version(dataset_handle, files, version_notes)
        else:
            raise (e)


def dataset_delete(owner_slug: str, dataset_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.post(
            f"/dataset/{owner_slug}/{dataset_slug}/delete",
            {},
        )
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Dataset '{dataset_slug}' for user '{owner_slug}'...")
        else:
            raise (e)


def internal_dataset_download(
    handle: str, path: Optional[str] = None, *, force_download: Optional[bool] = False, referrer: Optional[str] = None
) -> str:
    """Download dataset files, with extra options intended for internal kagglehub usage
    Args:
        handle: (string) the dataset handle
        path: (string) Optional path to a file within a dataset
        force_download: (bool) Optional flag to force download a dataset, even if it's cached
        referrer: (string) Optional string to denote the referrer for the download
    Returns:
        A string requesting the path to the requested dataset files.
    """

    h = parse_dataset_handle(handle)
    logger.info(f"Downloading Dataset: {h.to_url()} ...", extra={**EXTRA_CONSOLE_BLOCK})
    return registry.dataset_resolver(h, path, force_download=force_download, referrer=referrer)
