import logging
from http import HTTPStatus

from kagglesdk.datasets.types.dataset_api_service import (
    ApiCreateDatasetRequest,
    ApiCreateDatasetVersionRequest,
    ApiCreateDatasetVersionRequestBody,
    ApiDeleteDatasetRequest,
)

from kagglehub.clients import BackendError, build_kaggle_client
from kagglehub.exceptions import KaggleApiHTTPError, handle_mutate_call
from kagglehub.gcs_upload import UploadDirectoryInfo
from kagglehub.handle import DatasetHandle

logger = logging.getLogger(__name__)


def _create_dataset(dataset_handle: DatasetHandle, upload_dir: UploadDirectoryInfo) -> None:
    upload_proto = upload_dir.to_proto()

    with build_kaggle_client() as api_client:
        r = ApiCreateDatasetRequest()
        r.owner_slug = dataset_handle.owner
        r.slug = dataset_handle.dataset
        r.title = dataset_handle.dataset
        r.files = upload_proto.files
        r.directories = upload_proto.directories
        r.is_private = True
        handle_mutate_call(lambda: api_client.datasets.dataset_api_client.create_dataset(r))

        logger.info(f"Your dataset has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}")


def _create_dataset_version(
    dataset_handle: DatasetHandle, upload_dir: UploadDirectoryInfo, version_notes: str = ""
) -> None:
    upload_proto = upload_dir.to_proto()

    with build_kaggle_client() as api_client:
        r = ApiCreateDatasetVersionRequest()
        r.owner_slug = dataset_handle.owner
        r.dataset_slug = dataset_handle.dataset
        r.body = ApiCreateDatasetVersionRequestBody()
        r.body.version_notes = version_notes
        r.body.files = upload_proto.files
        r.body.directories = upload_proto.directories
        handle_mutate_call(lambda: api_client.datasets.dataset_api_client.create_dataset_version(r))
        logger.info(
            f"Your dataset version has been created.\nFiles are being processed...\nSee at: {dataset_handle.to_url()}"
        )


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


def delete_dataset(owner_slug: str, dataset_slug: str) -> None:
    try:
        with build_kaggle_client() as api_client:
            r = ApiDeleteDatasetRequest()
            r.owner_slug = owner_slug
            r.dataset_slug = dataset_slug
            handle_mutate_call(lambda: api_client.datasets.dataset_api_client.delete_dataset(r))
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Dataset '{dataset_slug}' for user '{owner_slug}'...")
        else:
            raise (e)
