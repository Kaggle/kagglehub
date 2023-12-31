from http import HTTPStatus
from typing import Any, Dict, Optional

import requests


class CredentialError(Exception):
    pass


class KaggleEnvironmentError(Exception):
    pass


class ColabEnvironmentError(Exception):
    pass


class BackendError(Exception):
    pass


class NotFoundError(Exception):
    pass


class DataCorruptionError(Exception):
    pass


class KaggleApiHTTPError(requests.HTTPError):
    def __init__(self, message: str, response: Optional[requests.Response] = None):
        super().__init__(message, response=response)


def kaggle_api_raise_for_status(response: requests.Response):
    """
    Wrapper around `response.raise_for_status()` that provides nicer error messages
    See: https://requests.readthedocs.io/en/latest/api/#requests.Response.raise_for_status
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        message = str(e)

        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            message = (
                f"{response.status_code} Client Error."
                "\n\n"
                f"You don't have permission to access resource at URL: {response.url}"
                "\nPlease make sure you are authenticated if you are trying to access a private resource or a resource"
                " requiring consent."
            )
        if response.status_code == HTTPStatus.NOT_FOUND:
            message = (
                f"{response.status_code} Client Error."
                "\n\n"
                f"Resource not found at URL: {response.url}"
                "\nPlease make sure you specified the correct resource identifiers."
            )

        # Default handling
        raise KaggleApiHTTPError(message, response=response) from e


def process_post_response(response: Dict[str, Any]):
    """
    Postprocesses the API response to check for errors.
    """
    if not (200 <= response.get("code", 200) < 300):  # noqa: PLR2004
        error_message = response.get("message", "No error message provided")
        raise BackendError(error_message)
    elif "error" in response and response["error"] != "":
        raise BackendError(response["error"])
