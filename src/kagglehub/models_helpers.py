import logging
from http import HTTPStatus
from typing import List, Optional

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError, postprocess_response
from kagglehub.handle import ModelHandle

logger = logging.getLogger(__name__)


def _create_model(owner_slug: str, model_slug: str):
    data = {"ownerSlug": owner_slug, "slug": model_slug, "title": model_slug, "isPrivate": True}
    api_client = KaggleApiV1Client()
    response = api_client.post("/models/create/new", data)
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    postprocess_response(response)
    logger.info("Model Created.")


def _create_model_instance(model_handle: ModelHandle, license_name: str, files: List[str]):
    data = {
        "instanceSlug": model_handle.variation,
        "framework": model_handle.framework,
        "licenseName": license_name,
        "overview": "test",
        "usage": "test",
        "trainingData": ["test"],
        "files": [{"token": files[0]}],
    }
    api_client = KaggleApiV1Client()
    response = api_client.post(f"/models/{model_handle.owner}/{model_handle.model}/create/instance", data)
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    postprocess_response(response)
    logger.info("Model Instance Created.")


def _create_model_instance_version(model_handle: ModelHandle, files: List[str], version_notes=""):
    data = {"versionNotes": version_notes, "files": [{"token": files[0]}]}
    api_client = KaggleApiV1Client()
    response = api_client.post(
        f"/models/{model_handle.owner}/{model_handle.model}/{model_handle.framework}/{model_handle.variation}/create/version",
        data,
    )
    # Note: The API doesn't throw on error. It returns 200 and you need to check the 'error' field.
    postprocess_response(response)
    logger.info("Model Instance Version Created.")


def create_model_instance_or_version(
    model_handle: ModelHandle, license_name: str, files: List[str], version_notes: Optional[str] = None
):
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{model_handle}/get")
        # the instance exist, create a new version.
        _create_model_instance_version(model_handle, files, version_notes)
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            _create_model_instance(model_handle, license_name, files)
        else:
            raise (e)


def get_or_create_model(owner_slug: str, model_slug: str):
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/models/{owner_slug}/{model_slug}/get")
    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Model '{model_slug}' does not exist for user '{owner_slug}'. Creating Model...")
            _create_model(owner_slug, model_slug)
        else:
            raise (e)
