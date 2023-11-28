import json
import logging
import os
from typing import Tuple
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

from kagglehub.config import get_kaggle_api_endpoint, get_kaggle_credentials
from kagglehub.exceptions import BackendError, CredentialError, KaggleEnvironmentError

CHUNK_SIZE = 1048576
# The `connect` timeout is the number of seconds `requests` will wait for your client to establish a connection.
# The `read` timeout is the number of seconds that the client will wait BETWEEN bytes sent from the server.
# See: https://requests.readthedocs.io/en/stable/user/advanced/#timeouts
DEFAULT_CONNECT_TIMEOUT = 5  # seconds
DEFAULT_READ_TIMEOUT = 15  # seconds
ACCEPT_RANGE_HTTP_HEADER = "Accept-Ranges"

logger = logging.getLogger(__name__)


# TODO(b/307576378): When ready, use `kagglesdk` to issue requests.
class KaggleApiV1Client:
    BASE_PATH = "api/v1"

    def __init__(self):
        self.credentials = get_kaggle_credentials()
        self.endpoint = get_kaggle_api_endpoint()

    def get(self, path: str) -> dict:
        url = self._build_url(path)
        with requests.get(
            url,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            response.raise_for_status()
            return response.json()

    def download_file(self, path: str, out_file: str):
        url = self._build_url(path)
        logger.info(f"Downloading from {url}...")
        with requests.get(
            url,
            stream=True,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            response.raise_for_status()
            total_size = int(response.headers["Content-Length"])
            size_read = 0

            if _is_resumable(response) and os.path.isfile(out_file):
                size_read = os.path.getsize(out_file)

                logger.info(f"Resuming download from {size_read} bytes ({total_size - size_read} bytes left)...")

                # Send the request again with the 'Range' header.
                with requests.get(
                    response.url,  # URL after redirection
                    stream=True,
                    auth=self._get_http_basic_auth(),
                    timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
                    headers={"Range": f"bytes={size_read}-"},
                ) as resumed_response:
                    _download_file(resumed_response, out_file, size_read, total_size)
            else:
                _download_file(response, out_file, size_read, total_size)

    def _get_http_basic_auth(self):
        if self.credentials:
            return HTTPBasicAuth(self.credentials.username, self.credentials.key)
        return None

    def _build_url(self, path: str):
        return urljoin(self.endpoint, f"{KaggleApiV1Client.BASE_PATH}/{path}")


def _is_resumable(response: requests.Response):
    return ACCEPT_RANGE_HTTP_HEADER in response.headers and response.headers[ACCEPT_RANGE_HTTP_HEADER] == "bytes"


def _download_file(response: requests.Response, out_file: str, size_read: int, total_size: int):
    open_mode = "ab" if size_read > 0 else "wb"
    with tqdm(total=total_size, initial=size_read, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
        with open(out_file, open_mode) as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
                size_read = min(total_size, size_read + CHUNK_SIZE)
                progress_bar.update(len(chunk))


# These environment variables are set by the Kaggle notebook environment.
KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME = "KAGGLE_DATA_PROXY_URL"
KAGGLE_JWT_TOKEN_ENV_VAR_NAME = "KAGGLE_USER_SECRETS_TOKEN"
KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME = "KAGGLE_DATA_PROXY_TOKEN"


class KaggleJwtClient:
    BASE_PATH = "/kaggle-jwt-handler/"

    def __init__(self):
        self.endpoint = os.getenv(KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME)
        if self.endpoint is None:
            msg = f"The {KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME} should be set."
            raise KaggleEnvironmentError(msg)
        jwt_token = os.getenv(KAGGLE_JWT_TOKEN_ENV_VAR_NAME)
        if jwt_token is None:
            msg = (
                "A JWT Token is required to call Kaggle, "
                f"but none found in environment variable {KAGGLE_JWT_TOKEN_ENV_VAR_NAME}"
            )
            raise CredentialError(msg)

        data_proxy_token = os.getenv(KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME)
        if data_proxy_token is None:
            msg = (
                "A Data Proxy Token is required to call Kaggle, "
                f"but none found in environment variable {KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME}"
            )
            raise CredentialError(msg)

        self.headers = {
            "Content-type": "application/json",
            "X-Kaggle-Authorization": f"Bearer {jwt_token}",
            "X-KAGGLE-PROXY-DATA": data_proxy_token,
        }

    def post(
        self,
        request_name: str,
        data: dict,
        timeout: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
    ) -> dict:
        url = f"{self.endpoint}{KaggleJwtClient.BASE_PATH}{request_name}"
        with requests.post(
            url,
            headers=self.headers,
            data=bytes(json.dumps(data), "utf-8"),
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            json_response = response.json()
            if "wasSuccessful" not in json_response:
                msg = "'wasSuccessful' field missing from response"
                raise BackendError(msg)
            if not json_response["wasSuccessful"]:
                msg = f"POST failed with: {response.text!s}"
                raise BackendError(msg)
            if "result" not in json_response:
                msg = "'result' field missing from response"
                raise BackendError(msg)
            return json_response["result"]
