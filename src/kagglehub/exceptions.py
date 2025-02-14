import logging
from http import HTTPStatus
from typing import Any, Optional

import requests

from kagglehub.handle import CompetitionHandle, ResourceHandle
from kagglehub.logger import EXTRA_CONSOLE_BLOCK

logger = logging.getLogger(__name__)


class CredentialError(Exception):
    pass


class KaggleEnvironmentError(Exception):
    pass


class ColabEnvironmentError(Exception):
    pass


class BackendError(Exception):
    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        self.error_code = error_code
        super().__init__(message)


class NotFoundError(Exception):
    pass


class DataCorruptionError(Exception):
    pass


class KaggleApiHTTPError(requests.HTTPError):
    def __init__(self, message: str, response: Optional[requests.Response] = None) -> None:
        super().__init__(message, response=response)


class ColabHTTPError(requests.HTTPError):
    def __init__(self, message: str, response: Optional[requests.Response] = None) -> None:
        super().__init__(message, response=response)


class UnauthenticatedError(Exception):
    """Exception raised for errors in the authentication process."""

    def __init__(self, message: str = "User is not authenticated") -> None:
        super().__init__(message)


class UserCancelledError(Exception):
    pass


def kaggle_api_raise_for_status(response: requests.Response, resource_handle: Optional[ResourceHandle] = None) -> None:
    """
    Wrapper around `response.raise_for_status()` that provides nicer error messages
    See: https://requests.readthedocs.io/en/latest/api/#requests.Response.raise_for_status
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        message = str(e)
        server_error_message = ""
        try:
            server_error_message = response.json().get("message", "")
            if server_error_message:
                server_error_message = f"The server reported the following issues: {server_error_message}\n"
        except requests.exceptions.JSONDecodeError as ex:
            logger.info(f"Server payload is not json. See {ex}", extra={**EXTRA_CONSOLE_BLOCK})
        resource_url = resource_handle.to_url() if resource_handle else response.url
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            if isinstance(resource_handle, CompetitionHandle):
                message = (
                    f"{response.status_code} Client Error."
                    "\n\n"
                    f"You don't have permission to access resource at URL: {resource_url}"
                    "\nPlease make sure you are authenticated and have accepted the competition rules which"
                    f" can be found at this location: {resource_url}/rules"
                )
            else:
                message = (
                    f"{response.status_code} Client Error."
                    "\n\n"
                    f"You don't have permission to access resource at URL: {resource_url}. "
                    f"{server_error_message}"
                    f"Please make sure you are authenticated if you are trying to access a "
                    f"private resource or a resource requiring consent."
                )

        if response.status_code == HTTPStatus.NOT_FOUND:
            message = (
                f"{response.status_code} Client Error."
                "\n\n"
                f"Resource not found at URL: {resource_url}\n"
                f"{server_error_message}"
                "Please make sure you specified the correct resource identifiers."
            )

        # Default handling
        raise KaggleApiHTTPError(message, response=response) from e


def colab_raise_for_status(response: requests.Response, resource_handle: Optional[ResourceHandle] = None) -> None:
    """
    Wrapper around `response.raise_for_status()` that provides nicer error messages
    See: https://requests.readthedocs.io/en/latest/api/#requests.Response.raise_for_status
    """
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        message = str(e)
        resource_url = resource_handle.to_url() if resource_handle else response.url

        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            message = (
                f"{response.status_code} Client Error."
                "\n\n"
                f"You don't have permission to access resource at URL: {resource_url}"
                "\nPlease make sure you are authenticated if you are trying to access a private resource or a resource"
                " requiring consent."
            )
        # Default handling
        raise ColabHTTPError(message, response=response) from e


def process_post_response(response: dict[str, Any]) -> None:
    """
    Postprocesses the API response to check for errors.
    """
    if not (200 <= response.get("code", 200) < 300):  # noqa: PLR2004
        error_message = response.get("message", "No error message provided")
        raise BackendError(error_message)
    elif "error" in response and response["error"] != "":
        error_code = int(response["errorCode"]) if "errorCode" in response else None
        raise BackendError(response["error"], error_code)
