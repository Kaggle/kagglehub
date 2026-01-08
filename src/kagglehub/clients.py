import hashlib
import inspect
import json
import logging
import os
import sys
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
import requests.auth
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv, get_env, is_in_kaggle_notebook
from kagglesdk.kaggle_http_client import KaggleHttpClient
from packaging.version import parse
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

import kagglehub
from kagglehub.cache import delete_from_cache, get_cached_archive_path
from kagglehub.config import get_kaggle_credentials
from kagglehub.datasets_enums import KaggleDatasetAdapter
from kagglehub.env import (
    KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME,
    is_in_colab_notebook,
    read_kaggle_build_date,
    search_lib_in_call_stack,
)
from kagglehub.exceptions import (
    BackendError,
    ColabEnvironmentError,
    CredentialError,
    DataCorruptionError,
    KaggleEnvironmentError,
    NotFoundError,
    colab_raise_for_status,
)
from kagglehub.handle import CompetitionHandle, ResourceHandle
from kagglehub.integrity import get_md5_checksum_from_response, to_b64_digest, update_hash_from_file

CHUNK_SIZE = 1048576
# The `connect` timeout is the number of seconds `requests` will wait for your client to establish a connection.
# The `read` timeout is the number of seconds that the client will wait BETWEEN bytes sent from the server.
# See: https://requests.readthedocs.io/en/stable/user/advanced/#timeouts
DEFAULT_CONNECT_TIMEOUT = 5  # seconds
DEFAULT_READ_TIMEOUT = 15  # seconds
ACCEPT_RANGE_HTTP_HEADER = "Accept-Ranges"
HTTP_STATUS_404 = 404

already_printed_version_warning = False

_CHECKSUM_MISMATCH_MSG_TEMPLATE = """\
The X-Goog-Hash header indicated a MD5 checksum of:

  {}

but the actual MD5 checksum of the downloaded contents was:

  {}
"""

ADAPTER_TO_USER_AGENT_MAP = {
    KaggleDatasetAdapter.HUGGING_FACE: "hugging_face_data_loader",
    KaggleDatasetAdapter.PANDAS: "pandas_data_loader",
    KaggleDatasetAdapter.POLARS: "polars_data_loader",
}


def get_user_agent() -> str:
    """Identifies the user agent based on available system information.

    Returns:
        str: user agent information.
    """
    user_agents = [f"kagglehub/{kagglehub.__version__}"]
    for lib in ("keras_hub", "keras_nlp", "keras_cv", "keras", "torchtune"):
        lib_info = search_lib_in_call_stack(lib)
        if lib_info is not None:
            user_agents.append(lib_info)
            break

    # Add an appropriate data loader user agent for kagglehub.dataset_load calls
    for frame_info in inspect.stack():
        if frame_info.function != "dataset_load":
            continue
        module = inspect.getmodule(frame_info.frame)

        if not module or not hasattr(module, "__name__") or not module.__name__.startswith("kagglehub.datasets"):
            continue

        # We've confirmed that this is a call to kagglehub.dataset_load. Now figure out which loader was used.
        adapter = frame_info.frame.f_locals["adapter"] if "adapter" in frame_info.frame.f_locals else None
        if adapter and adapter in ADAPTER_TO_USER_AGENT_MAP:
            user_agents.append(ADAPTER_TO_USER_AGENT_MAP[adapter])
            break

    if is_in_kaggle_notebook():
        build_date = read_kaggle_build_date()
        user_agents.append(f"kkb/{build_date}")
    elif is_in_colab_notebook():
        colab_tag = os.getenv("COLAB_RELEASE_TAG")
        runtime_suffix = "-managed" if os.getenv("TBE_RUNTIME_ADDR") else "-unmanaged"
        user_agents.append(f"colab/{colab_tag}{runtime_suffix}")

    return " ".join(user_agents)


def get_response_processor() -> Callable[..., None]:
    return _check_response_version


def _check_response_version(response: requests.Response) -> None:
    global already_printed_version_warning  # noqa: PLW0603
    if already_printed_version_warning:
        return
    latest_version_str = response.headers.get("X-Kaggle-HubVersion")
    if latest_version_str:
        current_version = parse(kagglehub.__version__)
        latest_version = parse(latest_version_str)
        if latest_version > current_version:
            sys.stdout.write(
                "Warning: Looks like you're using an outdated `kagglehub` "
                f"version (installed: {current_version}), please consider "
                f"upgrading to the latest version ({latest_version_str})"
            )
            already_printed_version_warning = True


logger = logging.getLogger(__name__)


def build_kaggle_client() -> KaggleClient:
    credentials = get_kaggle_credentials()
    env = get_env()
    verbose = True if env == KaggleEnv.TEST else False
    if not credentials:
        # Unauthenticated client
        return KaggleClient(
            env=env,
            verbose=verbose,
            user_agent=get_user_agent(),
        )

    return KaggleClient(
        env=env,
        verbose=verbose,
        username=credentials.username,
        password=credentials.key,
        api_token=credentials.api_key,
        user_agent=get_user_agent(),
        response_processor=get_response_processor(),
    )


def download_file(
    response: requests.Response,
    out_file: str,
    resource_handle: ResourceHandle,
    cached_path: str | None = None,
    *,
    extract_auto_compressed_file: bool = False,
) -> bool:
    """
    Issues a call to kaggle api and downloads files. For competition downloads,
    call may return early if local cache is newer than the last time the file was modified.

    Returns:
    bool:  If downloading remote was necessary
    """
    total_size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
    size_read = 0

    if isinstance(resource_handle, CompetitionHandle) and not _download_needed(response, resource_handle, cached_path):
        return False

    expected_md5_hash = get_md5_checksum_from_response(response)
    hash_object = hashlib.md5() if expected_md5_hash else None

    if _is_resumable(response) and total_size and os.path.isfile(out_file):
        size_read = os.path.getsize(out_file)
        update_hash_from_file(hash_object, out_file)

        if size_read == total_size:
            logger.info(f"Download already complete ({size_read} bytes).")
            return True

        logger.info(f"Resuming download from {size_read} bytes ({total_size - size_read} bytes left)...")

        # Send the request again with the 'Range' header.
        with requests.get(
            response.url,  # GCS URL after redirection
            stream=True,
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
            headers={"Range": f"bytes={size_read}-"},
        ) as resumed_response:
            logger.info(f"Resuming download to {out_file} ({size_read}/{total_size}) bytes left.")
            _download_file(resumed_response, out_file, size_read, total_size, hash_object)
    else:
        logger.info(f"Downloading to {out_file}...")
        _download_file(response, out_file, size_read, total_size, hash_object)

    if hash_object:
        actual_md5_hash = to_b64_digest(hash_object)
        if actual_md5_hash != expected_md5_hash:
            os.remove(out_file)  # Delete the corrupted file.
            raise DataCorruptionError(_CHECKSUM_MISMATCH_MSG_TEMPLATE.format(expected_md5_hash, actual_md5_hash))

    # For individual file downloads, the downloaded file may be a zip of the file rather
    # than the file name/type that was requested (e.g. my-big-table.csv.zip and not my-big-table.csv).
    # If that's the case, we should auto-extract it so users get what they expect.
    expected_downloaded_file_name = urlparse(out_file).path.split("/")[-1]
    actual_downloaded_file_name = urlparse(response.url).path.split("/")[-1]
    if (
        extract_auto_compressed_file
        and f"{expected_downloaded_file_name}.zip" == actual_downloaded_file_name
        and zipfile.is_zipfile(out_file)
    ):
        logger.info(f"Extracting zip of {expected_downloaded_file_name}...")
        # Rename the file to match what it really is and make space to write to the expected location
        renamed_auto_compressed_path = f"{out_file}.zip"
        os.rename(out_file, renamed_auto_compressed_path)
        with zipfile.ZipFile(renamed_auto_compressed_path, "r") as f:
            f.extract(expected_downloaded_file_name, os.path.dirname(out_file))
        # We don't need the zipped version anymore
        os.remove(renamed_auto_compressed_path)
    return True


def _is_resumable(response: requests.Response) -> bool:
    return ACCEPT_RANGE_HTTP_HEADER in response.headers and response.headers[ACCEPT_RANGE_HTTP_HEADER] == "bytes"


def _download_file(
    response: requests.Response,
    out_file: str,
    size_read: int,
    total_size: int | None,
    hash_object,  # noqa: ANN001 - no public type for hashlib hash
) -> None:
    open_mode = "ab" if size_read > 0 else "wb"
    if total_size is not None:
        with tqdm(total=total_size, initial=size_read, unit="B", unit_scale=True, unit_divisor=1024) as progress_bar:
            with open(out_file, open_mode) as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    f.write(chunk)
                    if hash_object:
                        hash_object.update(chunk)
                    size_read = min(total_size, size_read + CHUNK_SIZE)
                    progress_bar.update(len(chunk))
    else:
        with open(out_file, open_mode) as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)
                if hash_object:
                    hash_object.update(chunk)


def _download_needed(response: requests.Response, h: ResourceHandle, cached_path: str | None = None) -> bool:
    """
    Determine if a download is needed based on timestamp and cached path.

    Returns:
        bool: download needed.
    """
    if not cached_path:
        return True

    last_modified = response.headers.get("Last-Modified")
    if last_modified is None:
        delete_from_cache(h, cached_path)

        archive_path = get_cached_archive_path(h)
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)

        return True
    else:
        remote_date = datetime.strptime(response.headers["Last-Modified"], "%a, %d %b %Y %H:%M:%S %Z").replace(
            tzinfo=timezone.utc
        )

    file_exists = os.path.exists(cached_path)
    if file_exists:
        local_date = datetime.fromtimestamp(os.path.getmtime(cached_path), tz=timezone.utc)

        download_needed = remote_date >= local_date
        if download_needed:
            delete_from_cache(h, cached_path)

            archive_path = get_cached_archive_path(h)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)

            return True

        return False

    delete_from_cache(h, cached_path)

    archive_path = get_cached_archive_path(h)
    os.makedirs(os.path.dirname(archive_path), exist_ok=True)

    return True


# These environment variables are set by the Kaggle notebook environment.
KAGGLE_JWT_TOKEN_ENV_VAR_NAME = "KAGGLE_USER_SECRETS_TOKEN"
KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME = "KAGGLE_DATA_PROXY_TOKEN"


class KaggleJwtClient:
    BASE_PATH = "/kaggle-jwt-handler/"

    def __init__(self) -> None:
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
        timeout: tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
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
    IS_MODEL_SUPPORTED_PATH = "/kagglehub/models/is_supported"
    IS_DATASET_SUPPORTED_PATH = "/kagglehub/datasets/is_supported"
    MOUNT_PATH = "/kagglehub/models/mount"
    MODEL_MOUNT_PATH = "/kagglehub/models/mount"
    DATASET_MOUNT_PATH = "/kagglehub/datasets/mount"
    # TBE_RUNTIME_ADDR serves requests made from `is_supported` and  `_resolve`
    # of ModelColabCacheResolver.
    TBE_RUNTIME_ADDR_ENV_VAR_NAME = "TBE_RUNTIME_ADDR"

    def __init__(self) -> None:
        self.endpoint = os.getenv(self.TBE_RUNTIME_ADDR_ENV_VAR_NAME)
        if self.endpoint is None:
            msg = f"The {self.TBE_RUNTIME_ADDR_ENV_VAR_NAME} should be set."
            raise ColabEnvironmentError(msg)

        self.credentials = get_kaggle_credentials()
        self.headers = {"Content-type": "application/json"}

    def post(self, data: dict, handle_path: str, resource_handle: ResourceHandle | None = None) -> dict | None:
        url = f"http://{self.endpoint}{handle_path}"
        with requests.post(
            url,
            data=json.dumps(data),
            auth=self._get_auth(),
            headers=self.headers,
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as response:
            if response.status_code == HTTP_STATUS_404:
                raise NotFoundError()
            colab_raise_for_status(response, resource_handle)
            if response.text:
                return response.json()
        return None

    def _get_auth(self) -> requests.auth.AuthBase | None:
        if self.credentials:
            if self.credentials.username and self.credentials.key:
                return HTTPBasicAuth(self.credentials.username, self.credentials.key)
            if self.credentials.api_key:
                return KaggleHttpClient.BearerAuth(self.credentials.api_key)
        return None
