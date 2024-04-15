import logging
from http import HTTPStatus
from typing import List

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import DatasetHandle

logger = logging.getLogger(__name__)


def _create_dataset(dataset_handle: DatasetHandle, files: List[str]) -> None:
    data = {"files": [{"token": file_token} for file_token in files]}
    api_client = KaggleApiV1Client()
    api_client.post("/datasets/create/new", data)
    logger.info(
        f"Your dataset instance has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}"
    )


def _create_dataset_version(dataset_handle: DatasetHandle, files: List[str], version_notes: str = "") -> None:
    data = {"versionNotes": version_notes, "files": [{"token": file_token} for file_token in files]}
    api_client = KaggleApiV1Client()
    api_client.post(f"/datasets/create/{dataset_handle.version}/{dataset_handle.owner}/{dataset_handle.dataset}", data)
    logger.info(f"Your dataset has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}")


def create_dataset_or_version(dataset_handle: DatasetHandle, files: List[str], version_notes: str = "") -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/view/datasets/{dataset_handle}", dataset_handle)
        # the instance exists, create a new version.
        _create_dataset_version(dataset_handle, files, version_notes)
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
            _create_dataset(dataset_handle, files)
        else:
            raise (e)


def delete_dataset(owner_slug: str, dataset_slug: str) -> None:
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
