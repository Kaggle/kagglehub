import logging
from http import HTTPStatus
from typing import List, Optional, Callable

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import ModelHandle
from kagglehub.tracing import TraceContext

logger = logging.getLogger(__name__)


def _create_model(owner_slug: str, model_slug: str,
                  ctx_factory: Callable[[], TraceContext] = None) -> None:
    data = {
        "ownerSlug": owner_slug,
        "slug": model_slug,
        "title": model_slug,
        "isPrivate": True
    }
    api_client = KaggleApiV1Client(ctx_factory)
    api_client.post("/models/create/new", data)
    logger.info(f"Model '{model_slug}' Created.")


def _create_model_instance(model_handle: ModelHandle, files: List[str], license_name: Optional[str] = None,
                           ctx_factory: Callable[[], TraceContext] = None) -> None:
    data = {
        "instanceSlug": model_handle.variation,
        "framework": model_handle.framework,
        "files": [{"token": file_token} for file_token in files],
    }
    if license_name is not None:
        data["licenseName"] = license_name

    api_client = KaggleApiV1Client(ctx_factory)
    api_client.post(
        f"/models/{model_handle.owner}/{model_handle.model}/create/instance", data)
    logger.info(f"Your model instance has been created.\nFiles are being processed...\nSee at: {
                model_handle.to_url()}")


def _create_model_instance_version(model_handle: ModelHandle, files: List[str], version_notes: str = "",
                                   ctx_factory: Callable[[], TraceContext] = None) -> None:
    data = {
        "versionNotes": version_notes,
        "files": [
            {"token": file_token} for file_token in files]
    }
    api_client = KaggleApiV1Client(ctx_factory)
    api_client.post(
        f"/models/{model_handle.owner}/{model_handle.model}/{
            model_handle.framework}/{model_handle.variation}/create/version",
        data,
    )
    logger.info(
        f"Your model instance version has been created.\nFiles are being processed...\nSee at: {
            model_handle.to_url()}"
    )


def create_model_instance_or_version(
    model_handle: ModelHandle, files: List[str], license_name: Optional[str], version_notes: str = "",
    ctx_factory: Callable[[], TraceContext] = None
) -> None:
    try:
        api_client = KaggleApiV1Client(ctx_factory)
        api_client.get(f"/models/{model_handle}/get", model_handle)
        # the instance exist, create a new version.
        _create_model_instance_version(
            model_handle, files, version_notes, ctx_factory)
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
            _create_model_instance(model_handle, files,
                                   license_name, ctx_factory)
        else:
            raise (e)


def create_model_if_missing(owner_slug: str, model_slug: str,
                            ctx_factory: Callable[[], TraceContext] = None) -> None:
    try:
        api_client = KaggleApiV1Client(ctx_factory)
        api_client.get(f"/models/{owner_slug}/{model_slug}/get")
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
            logger.info(
                f"Model '{model_slug}' does not exist or access is forbidden for user '{owner_slug}'. Creating or handling Model..."  # noqa: E501
            )
            _create_model(owner_slug, model_slug, ctx_factory)
        else:
            raise (e)


def delete_model(owner_slug: str, model_slug: str,
                 ctx_factory: Callable[[], TraceContext] = None) -> None:
    try:
        api_client = KaggleApiV1Client(ctx_factory)
        api_client.post(
            f"/models/{owner_slug}/{model_slug}/delete",
            {},
        )
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not delete Model '{
                        model_slug}' for user '{owner_slug}'...")
        else:
            raise (e)
