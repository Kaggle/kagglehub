import logging
from http import HTTPStatus
<<<<<<< HEAD
from typing import List

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.gcs_upload import UploadDirectoryInfo
=======
from typing import List, Optional

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
>>>>>>> 66db1ff (add helpers/other files  and modify tests)
from kagglehub.handle import DatasetHandle

logger = logging.getLogger(__name__)

<<<<<<< HEAD

def _create_dataset(dataset_handle: DatasetHandle, files_and_directories: UploadDirectoryInfo) -> None:
    serialized_data = files_and_directories.serialize()
    data = {
        "ownerSlug": dataset_handle.owner,
        "datasetSlug": dataset_handle.dataset,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
        "directories": serialized_data["directories"],
    }

    api_client = KaggleApiV1Client()
    api_client.post("/datasets/create/new", data)
    logger.info(
        f"Your dataset instance has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}"
    )


def _create_dataset_version(dataset_handle: DatasetHandle, files_and_directories: UploadDirectoryInfo, version_notes: str = "") -> None:
    serialized_data = files_and_directories.serialize()
    data = {
        "versionNotes": version_notes,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
        "directories": serialized_data["directories"],
    }
    api_client = KaggleApiV1Client()
    api_client.post(f"/datasets/create/{dataset_handle.version}/{dataset_handle.owner}/{dataset_handle.dataset}", data)
    logger.info(f"Your dataset has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}")


def create_dataset_or_version(dataset_handle: DatasetHandle, files: UploadDirectoryInfo, version_notes: str = "") -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/view/datasets/{dataset_handle}", dataset_handle)
        # the instance exists, create a new version.
        _create_dataset_version(dataset_handle, files, version_notes)
=======
def _create_dataset(owner_slug: str, dataset_slug: str) -> None:
    data = {"ownerSlug": owner_slug, "slug": dataset_slug, "title": dataset_slug, "isPrivate": True}
    api_client = KaggleApiV1Client()
    api_client.post("/datasets/create/new", data)
    logger.info(f"Dataset '{dataset_slug}' Created.")

def create_dataset_if_missing(owner_slug: str, dataset_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/datasets/{owner_slug}/{dataset_slug}/get")
>>>>>>> 66db1ff (add helpers/other files  and modify tests)
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
<<<<<<< HEAD
            _create_dataset(dataset_handle, files)
        else:
            raise (e)


def delete_dataset(owner_slug: str, dataset_slug: str) -> None:
=======
            logger.info(
                f"Model '{dataset_slug}' does not exist or access is forbidden for user '{owner_slug}'. Creating or handling Dataset..."  # noqa: E501
            )
            _create_dataset(owner_slug, dataset_slug)
        else:
            raise (e)
        
def deleet_dataset(owner_slug: str, dataset_slug: str) -> None:
>>>>>>> 66db1ff (add helpers/other files  and modify tests)
    try:
        api_client = KaggleApiV1Client()
        api_client.post(
            f"/datasets/{owner_slug}/{dataset_slug}/delete",
            {},
        )
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Dataset '{dataset_slug}' for user '{owner_slug}'...")
        else:
            raise (e)
