import logging
from http import HTTPStatus

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import KaggleApiHTTPError

logger = logging.getLogger(__name__)


def is_notebook_utility_script(user_name: str, notebook_slug: str) -> None:
    try:
        api_client = KaggleApiV1Client()
        api_client.get(f"/kernels/pull/{user_name}/{notebook_slug}")

    except KaggleApiHTTPError as e:
        if e.response is not None and e.response.status_code == HTTPStatus.NOT_FOUND:
            logger.info(f"Could not find '{user_name}' for user '{notebook_slug}'...")
        else:
            raise (e)
