import logging
from http import HTTPStatus

from kagglesdk.models.types.model_api_service import (
    ApiCreateModelInstanceRequest,
    ApiCreateModelInstanceRequestBody,
    ApiCreateModelInstanceVersionRequest,
    ApiCreateModelInstanceVersionRequestBody,
    ApiCreateModelRequest,
    ApiDeleteModelRequest,
    ApiGetModelRequest,
    CreateModelSigningTokenRequest,
)

from kagglehub.clients import BackendError, build_kaggle_client
from kagglehub.exceptions import KaggleApiHTTPError, handle_call, handle_mutate_call
from kagglehub.gcs_upload import UploadDirectoryInfo
from kagglehub.handle import ModelHandle

logger = logging.getLogger(__name__)


def _create_model(owner_slug: str, model_slug: str) -> None:
    with build_kaggle_client() as api_client:
        r = ApiCreateModelRequest()
        r.owner_slug = owner_slug
        r.slug = model_slug
        r.title = model_slug
        r.is_private = True
        handle_mutate_call(lambda: api_client.models.model_api_client.create_model(r))
        logger.info(f"Model '{model_slug}' Created.")


def _create_model_instance(
    model_handle: ModelHandle,
    upload_dir: UploadDirectoryInfo,
    license_name: str | None = None,
    *,
    sigstore: bool | None = False,
) -> None:
    upload_proto = upload_dir.to_proto()

    with build_kaggle_client() as api_client:
        r = ApiCreateModelInstanceRequest()
        r.owner_slug = model_handle.owner
        r.model_slug = model_handle.model
        r.body = ApiCreateModelInstanceRequestBody()
        r.body.instance_slug = model_handle.variation
        r.body.framework = model_handle.framework_enum()
        r.body.files = upload_proto.files
        r.body.directories = upload_proto.directories
        r.body.sigstore = sigstore
        if license_name is not None:
            r.body.license_name = license_name
        handle_mutate_call(lambda: api_client.models.model_api_client.create_model_instance(r))

        logger.info(
            f"Your model instance has been created.\nFiles are being processed...\nSee at: {model_handle.to_url()}"
        )


def _create_model_instance_version(
    model_handle: ModelHandle,
    upload_dir: UploadDirectoryInfo,
    version_notes: str = "",
    *,
    sigstore: bool | None = False,
) -> None:
    upload_proto = upload_dir.to_proto()

    with build_kaggle_client() as api_client:
        r = ApiCreateModelInstanceVersionRequest()
        r.owner_slug = model_handle.owner
        r.model_slug = model_handle.model
        r.framework = model_handle.framework_enum()
        r.instance_slug = model_handle.variation
        r.body = ApiCreateModelInstanceVersionRequestBody()
        r.body.version_notes = version_notes
        r.body.files = upload_proto.files
        r.body.directories = upload_proto.directories
        r.body.sigstore = sigstore
        handle_mutate_call(lambda: api_client.models.model_api_client.create_model_instance_version(r))
        logger.info(
            f"Your model instance version has been created.\n"
            f"Files are being processed...\nSee at: {model_handle.to_url()}"
        )


def create_model_instance_or_version(
    model_handle: ModelHandle,
    files: UploadDirectoryInfo,
    license_name: str | None,
    version_notes: str = "",
    *,
    sigstore: bool | None = False,
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
        with build_kaggle_client() as api_client:
            r = ApiGetModelRequest()
            r.owner_slug = owner_slug
            r.model_slug = model_slug
            handle_call(lambda: api_client.models.model_api_client.get_model(r))
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
    except Exception as e:
        raise e


def delete_model(owner_slug: str, model_slug: str) -> None:
    try:
        with build_kaggle_client() as api_client:
            r = ApiDeleteModelRequest()
            r.owner_slug = owner_slug
            r.model_slug = model_slug
            handle_mutate_call(lambda: api_client.models.model_api_client.delete_model(r))
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Model '{model_slug}' for user '{owner_slug}'...")
        else:
            raise (e)


def signing_token(owner_slug: str, model_slug: str) -> str | None:
    "Returns a JWT for signing if authorized for /{owner_slug}/{model_slug}"
    try:
        with build_kaggle_client() as api_client:
            r = CreateModelSigningTokenRequest()
            r.owner_slug = owner_slug
            r.model_slug = model_slug
            response = handle_call(lambda: api_client.models.model_api_client.create_model_signing_token(r))
            return response.id_token
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(
                f"Could not get Signing token for Model '{model_slug}' for user '{owner_slug}'. Skipping signing..."
            )
        return ""
