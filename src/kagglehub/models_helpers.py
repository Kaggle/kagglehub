import logging
from http import HTTPStatus
from typing import Optional

from kagglehub.clients import BackendError, KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.gcs_upload import UploadDirectoryInfo
from kagglehub.handle import ModelHandle

logger = logging.getLogger(__name__)


def _create_model(owner_slug: str, model_slug: str) -> None:
    data = {"ownerSlug": owner_slug, "slug": model_slug, "title": model_slug, "isPrivate": True}
    api_client = KaggleApiV1Client()
    api_client.post("/models/create/new", data)
    logger.info(f"Model '{model_slug}' Created.")


def _create_model_instance(
    model_handle: ModelHandle,
    files_and_directories: UploadDirectoryInfo,
    license_name: Optional[str] = None,
    *,
    sigstore: Optional[bool] = False,
) -> None:
    serialized_data = files_and_directories.serialize()
    data = {
        "instanceSlug": model_handle.variation,
        "framework": model_handle.framework,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
        "directories": serialized_data["directories"],
        "sigstore": sigstore,
    }
    if license_name is not None:
        data["licenseName"] = license_name

    api_client = KaggleApiV1Client()
    api_client.post(f"/models/{model_handle.owner}/{model_handle.model}/create/instance", data)
    logger.info(f"Your model instance has been created.\nFiles are being processed...\nSee at: {model_handle.to_url()}")


def _create_model_instance_version(
    model_handle: ModelHandle,
    files_and_directories: UploadDirectoryInfo,
    version_notes: str = "",
    *,
    sigstore: Optional[bool] = False,
) -> None:
    serialized_data = files_and_directories.serialize()
    data = {
        "versionNotes": version_notes,
        "files": [{"token": file_token} for file_token in files_and_directories.files],
        "directories": serialized_data["directories"],
        "sigstore": sigstore,
    }
    api_client = KaggleApiV1Client()
    api_client.post(
        f"/models/{model_handle.owner}/{model_handle.model}/{model_handle.framework}/{model_handle.variation}/create/version",
        data,
    )
    logger.info(
        f"Your model instance version has been created.\nFiles are being processed...\nSee at: {model_handle.to_url()}"
    )


def create_model_instance_or_version(
    model_handle: ModelHandle,
    files: UploadDirectoryInfo,
    license_name: Optional[str],
    version_notes: str = "",
    *,
    sigstore: Optional[bool] = False,
) -> None:
    try:
        _create_model_instance(model_handle, files, license_name, sigstore=sigstore)
    except BackendError as e:
        if e.error_code == HTTPStatus.CONFLICT:
            # Instance already exist, creating a new version instead.
            _create_model_instance_version(model_handle, files, version_notes, sigstore=sigstore)
        else:
            raise (e)


def create_model_if_missing(owner_slug: str, model_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{owner_slug}/{model_slug}/get")
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
            logger.info(
                f"Model '{model_slug}' does not exist or access is forbidden for user '{owner_slug}'. Creating or handling Model..."  # noqa: E501
            )
            _create_model(owner_slug, model_slug)
        else:
            raise (e)


def delete_model(owner_slug: str, model_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.post(
            f"/models/{owner_slug}/{model_slug}/delete",
            {},
        )
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Model '{model_slug}' for user '{owner_slug}'...")
        else:
            raise (e)


def signing_token(owner_slug: str, model_slug: str) -> Optional[str]:
    "Returns a JWT for signing if authorized for /{owner_slug}/{model_slug}"
    try:
        api_client = KaggleApiV1Client()
        resp = api_client.post("/models/signing/token", {"ownerSlug": owner_slug, "modelSlug": model_slug})
        return resp.get("id_token")
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(
                f"Could not get Signing token for Model '{model_slug}' for user '{owner_slug}'. Skipping signing..."
            )
        return ""
