import logging
from http import HTTPStatus
from typing import List, Optional

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import DatasetHandle

logger = logging.getLogger(__name__)

def _create_dataset(owner_slug: str, dataset_slug: str) -> None:
    data = {"ownerSlug": owner_slug, "slug": dataset_slug, "title": dataset_slug, "isPrivate": True}
    api_client = KaggleApiV1Client()
    api_client.post("/datasets/create/new", data)
    logger.info(f"Dataset '{dataset_slug}' Created.")

def create_dataset_if_missing(owner_slug: str, dataset_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/datasets/{owner_slug}/{dataset_slug}/get")
    except KaggleApiHTTPError as e:
        if e.response is not None and (
            e.response.status_code == HTTPStatus.NOT_FOUND  # noqa: PLR1714
            or e.response.status_code == HTTPStatus.FORBIDDEN
        ):
            logger.info(
                f"Model '{dataset_slug}' does not exist or access is forbidden for user '{owner_slug}'. Creating or handling Dataset..."  # noqa: E501
            )
            _create_dataset(owner_slug, dataset_slug)
        else:
            raise (e)
        
def deleet_dataset(owner_slug: str, dataset_slug: str) -> None:
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
