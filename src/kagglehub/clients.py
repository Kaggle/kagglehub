import hashlib
import json
import logging
import os
from typing import Tuple
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

import kagglehub
from kagglehub.config import get_kaggle_api_endpoint, get_kaggle_credentials
from kagglehub.exceptions import (
    BackendError,
    ColabEnvironmentError,
    CredentialError,
    DataCorruptionError,
    KaggleEnvironmentError,
    NotFoundError,
    kaggle_api_raise_for_status,
    process_post_response,
)
from kagglehub.integrity import get_md5_checksum_from_response, to_b64_digest, update_hash_from_file

CHUNK_SIZE = 1048576
# The `connect` timeout is the number of seconds `requests` will wait for your client to establish a connection.
# The `read` timeout is the number of seconds that the client will wait BETWEEN bytes sent from the server.
# See: https://requests.readthedocs.io/en/stable/user/advanced/#timeouts
DEFAULT_CONNECT_TIMEOUT = 5  # seconds
DEFAULT_READ_TIMEOUT = 15  # seconds
ACCEPT_RANGE_HTTP_HEADER = "Accept-Ranges"
HTTP_STATUS_404 = 404

_CHECKSUM_MISMATCH_MSG_TEMPLATE = """\
The X-Goog-Hash header indicated a MD5 checksum of:

  {}

but the actual MD5 checksum of the downloaded contents was:

  {}
"""

KAGGLEHUB_USER_AGENT = {"User-Agent": f"kagglehub/{kagglehub.__version__}"}

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
            headers=KAGGLEHUB_USER_AGENT,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            kaggle_api_raise_for_status(response)
            return response.json()

    def post(self, path: str, data: dict):
        url = self._build_url(path)
        with requests.post(
            url,
            headers=KAGGLEHUB_USER_AGENT,
            json=data,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            response_dict = response.json()
            process_post_response(response_dict)
            return response_dict

    def download_file(self, path: str, out_file: str):
        url = self._build_url(path)
        logger.info(f"Downloading from {url}...")
        with requests.get(
            url,
            headers=KAGGLEHUB_USER_AGENT,
            stream=True,
            auth=self._get_http_basic_auth(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            kaggle_api_raise_for_status(response)
            total_size = int(response.headers["Content-Length"])
            size_read = 0

            expected_md5_hash = get_md5_checksum_from_response(response)
            hash_object = hashlib.md5() if expected_md5_hash else None

            if _is_resumable(response) and os.path.isfile(out_file):
                size_read = os.path.getsize(out_file)
                update_hash_from_file(hash_object, out_file)

                if size_read == total_size:
                    logger.info(f"Download already complete ({size_read} bytes).")
                    return

                logger.info(f"Resuming download from {size_read} bytes ({total_size - size_read} bytes left)...")

                # Send the request again with the 'Range' header.
                with requests.get(
                    response.url,  # URL after redirection
                    stream=True,
                    auth=self._get_http_basic_auth(),
                    timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
                    headers={"Range": f"bytes={size_read}-"},
                ) as resumed_response:
                    _download_file(resumed_response, out_file, size_read, total_size, hash_object)
            else:
                _download_file(response, out_file, size_read, total_size, hash_object)

            if hash_object:
                actual_md5_hash = to_b64_digest(hash_object)
                if actual_md5_hash != expected_md5_hash:
                    os.remove(out_file)  # Delete the corrupted file.
                    raise DataCorruptionError(
                        _CHECKSUM_MISMATCH_MSG_TEMPLATE.format(expected_md5_hash, actual_md5_hash)
                    )

    def _get_http_basic_auth(self):
        if self.credentials:
            return HTTPBasicAuth(self.credentials.username, self.credentials.key)
        return None

    def _build_url(self, path: str):
        return urljoin(self.endpoint, f"{KaggleApiV1Client.BASE_PATH}/{path}")


def _is_resumable(response: requests.Response):
    return ACCEPT_RANGE_HTTP_HEADER in response.headers and response.headers[ACCEPT_RANGE_HTTP_HEADER] == "bytes"


def _download_file(response: requests.Response, out_file: str, size_read: int, total_size: int, hash_object):
    open_mode = "ab" if size_read > 0 else "wb"
    with tqdm(total=total_size, initial=size_read, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
        with open(out_file, open_mode) as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
                if hash_object:
                    hash_object.update(chunk)
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


class ColabClient:
    IS_SUPPORTED_PATH = "/kagglehub/models/is_supported"
    MOUNT_PATH = "/kagglehub/models/mount"
    # TBE_RUNTIME_ADDR serves requests made from `is_supported` and  `__call__`
    # of ModelColabCacheResolver.
    TBE_RUNTIME_ADDR_ENV_VAR_NAME = "TBE_RUNTIME_ADDR"

    def __init__(self):
        self.endpoint = os.getenv(self.TBE_RUNTIME_ADDR_ENV_VAR_NAME)
        if self.endpoint is None:
            msg = f"The {self.TBE_RUNTIME_ADDR_ENV_VAR_NAME} should be set."
            raise ColabEnvironmentError(msg)

        self.credentials = get_kaggle_credentials()
        self.headers = {"Content-type": "application/json"}

    def post(self, data: dict, handle_path):
        url = f"http://{self.endpoint}{handle_path}"
        with requests.post(
            url,
            data=json.dumps(data),
            auth=self._get_http_basic_auth(),
            headers=self.headers,
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            if response.status_code == HTTP_STATUS_404:
                raise NotFoundError()
            response.raise_for_status()
            if response.text:
                return response.json()

    def _get_http_basic_auth(self):
        if self.credentials:
            return HTTPBasicAuth(self.credentials.username, self.credentials.key)
        return None
