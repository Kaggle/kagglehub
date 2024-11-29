import fnmatch
import logging
import os
import pathlib
import time
import zipfile
from collections.abc import Iterable, Sequence
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Optional, Union

import requests
from requests.exceptions import Timeout
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import BackendError

logger = logging.getLogger(__name__)

MAX_FILES_TO_UPLOAD = 50
TEMP_ARCHIVE_FILE = "archive.zip"
MAX_RETRIES = 5
REQUEST_TIMEOUT = 600


class UploadDirectoryInfo:
    def __init__(
        self,
        name: str,
        files: Optional[list[str]] = None,
        directories: Optional[list["UploadDirectoryInfo"]] = None,
    ):
        self.name = name
        self.files = files if files is not None else []
        self.directories = directories if directories is not None else []

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "files": [{"token": file} for file in self.files],
            "directories": [directory.serialize() for directory in self.directories],
        }


def parse_datetime_string(string: str) -> Union[datetime, str]:
    time_formats = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"]
    for t in time_formats:
        try:
            return datetime.strptime(string[:26], t).replace(microsecond=0)  # noqa: DTZ007
        except:  # noqa: E722, S110
            pass
    return string


class File(object):  # noqa: UP004
    def __init__(self, init_dict: dict) -> None:
        parsed_dict = {k: parse_datetime_string(v) for k, v in init_dict.items()}
        self.__dict__.update(parsed_dict)

    @staticmethod
    def get_size(size: float, precision: int = 0) -> str:
        suffixes = ["B", "KB", "MB", "GB", "TB"]
        suffix_index = 0
        while size >= 1024 and suffix_index < 4:  # noqa: PLR2004
            suffix_index += 1
            size /= 1024.0
        return f"{size:.{precision}f}{suffixes[suffix_index]}"


def filtered_walk(*, base_dir: str, ignore_patterns: Sequence[str]) -> Iterable[tuple[str, list[str], list[str]]]:
    """An `os.walk` like directory tree generator with filtering.

    This method filters out files matching any ignore pattern.

    Args:
        base_dir (str): The base dir to walk in.
        ignore_patterns (Sequence[str]):
            The patterns for ignored files. These are standard wildcards relative to base_dir.

    Yields:
        Iterable[tuple[str, list[str], list[str]]]: (base_dir_path, list[dir_names], list[filtered_file_names])
    """
    for dir_path, dir_names, file_names in os.walk(base_dir):
        dir_p = pathlib.Path(dir_path)
        filtered_files = []
        for file_name in file_names:
            rel_file_p = (dir_p / file_name).relative_to(base_dir)
            if not any(fnmatch.fnmatch(name=str(rel_file_p), pat=pat) for pat in ignore_patterns):
                filtered_files.append(file_name)
        if filtered_files:
            yield (dir_path, dir_names, filtered_files)


def _check_uploaded_size(session_uri: str, file_size: int, backoff_factor: int = 1) -> int:
    """Check the status of the resumable upload."""
    headers = {"Content-Length": "0", "Content-Range": f"bytes */{file_size}"}
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            response = requests.put(session_uri, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 308:  # Resume Incomplete # noqa: PLR2004
                range_header = response.headers.get("Range")
                if range_header:
                    bytes_uploaded = int(range_header.split("-")[1]) + 1
                    return bytes_uploaded
                return 0  # If no Range header, assume no bytes were uploaded
            else:
                return file_size
        except (requests.ConnectionError, Timeout):
            logger.info(f"Network issue while checking uploaded size, retrying in {backoff_factor} seconds...")
            time.sleep(backoff_factor)
            backoff_factor = min(backoff_factor * 2, 60)
            retry_count += 1

    return 0  # Return 0 if all retries fail


def _upload_blob(file_path: str, item_type: str) -> str:
    """Uploads a file to a remote server as a blob and returns an upload token.

    Args:
        file_path: The path to the file to be uploaded.
        item_type : The type of the item associated with the file.

    Returns:
        A str token of uploaded blob.
    """
    file_size = os.path.getsize(file_path)
    data = {
        "type": item_type,
        "name": os.path.basename(file_path),
        "contentLength": file_size,
        "lastModifiedEpochSeconds": int(os.path.getmtime(file_path)),
    }
    api_client = KaggleApiV1Client()
    response = api_client.post("/blobs/upload", data=data)

    # Validate response content
    if "createUrl" not in response:
        create_url_exception = "'createUrl' field missing from response"
        raise BackendError(create_url_exception)
    if "token" not in response:
        token_exception = "'token' field missing from response"
        raise BackendError(token_exception)

    session_uri = response["createUrl"]
    headers = {"Content-Type": "application/octet-stream"}

    retry_count = 0
    uploaded_bytes = 0
    backoff_factor = 1  # Initial backoff duration in seconds

    with open(file_path, "rb") as f, tqdm(total=file_size, desc="Uploading", unit="B", unit_scale=True) as pbar:
        while retry_count < MAX_RETRIES and (file_size == 0 or uploaded_bytes < file_size):
            try:
                # Special case for empty files.
                if file_size == 0:
                    headers["Content-Length"] = "0"
                    upload_data = None
                # Resumable upload for non-empty files.
                else:
                    f.seek(uploaded_bytes)
                    headers["Content-Range"] = f"bytes {uploaded_bytes}-{file_size - 1}/{file_size}"
                    upload_data = CallbackIOWrapper(pbar.update, f, "read")

                upload_response = requests.put(session_uri, headers=headers, data=upload_data, timeout=REQUEST_TIMEOUT)

                if upload_response.status_code in [200, 201]:
                    return response["token"]
                elif upload_response.status_code == 308:  # Resume Incomplete # noqa: PLR2004
                    uploaded_bytes = _check_uploaded_size(session_uri, file_size)
                else:
                    upload_failed_exception = (
                        f"Upload failed with status code {upload_response.status_code}: {upload_response.text}"
                    )
                    raise BackendError(upload_failed_exception)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.info(f"Network issue: {e}, retrying in {backoff_factor} seconds...")
                time.sleep(backoff_factor)
                backoff_factor = min(backoff_factor * 2, 60)
                retry_count += 1
                uploaded_bytes = _check_uploaded_size(session_uri, file_size)
                pbar.n = uploaded_bytes  # Update progress bar to reflect actual uploaded bytes

    return response["token"]


def upload_files_and_directories(
    folder: str,
    *,
    ignore_patterns: Sequence[str],
    item_type: str,
    quiet: bool = False,
) -> UploadDirectoryInfo:
    # Count the total number of files
    file_count = 0
    for _, _, files in filtered_walk(base_dir=folder, ignore_patterns=ignore_patterns):
        file_count += len(files)

    if file_count > MAX_FILES_TO_UPLOAD:
        if not quiet:
            logger.info(f"More than {MAX_FILES_TO_UPLOAD} files detected, creating a zip archive...")

        with TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, TEMP_ARCHIVE_FILE)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for root, _, files in filtered_walk(base_dir=folder, ignore_patterns=ignore_patterns):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder))

            tokens = [
                token
                for token in [_upload_file(file_path=zip_path, item_type=item_type, quiet=quiet)]
                if token is not None
            ]
            return UploadDirectoryInfo(name="archive", files=tokens)

    root_dict = UploadDirectoryInfo(name="root")
    if os.path.isfile(folder):
        # Directly upload the file if the path is a file
        token = _upload_file(file_path=folder, item_type=item_type, quiet=quiet)
        if token:
            root_dict.files.append(token)
    else:
        for root, _, files in filtered_walk(base_dir=folder, ignore_patterns=ignore_patterns):
            # Path of the current folder relative to the base folder
            path = os.path.relpath(root, folder)

            # Navigate or create the dictionary path to the current folder
            current_dict = root_dict
            if path != ".":
                for part in path.split(os.sep):
                    # Find or create the subdirectory in the current dictionary
                    for subdir in current_dict.directories:
                        if subdir.name == part:
                            current_dict = subdir
                            break
                    else:
                        # If the directory is not found, create a new one
                        new_dir = UploadDirectoryInfo(name=part)
                        current_dict.directories.append(new_dir)
                        current_dict = new_dir

            # Add file tokens to the current directory in the dictionary
            for file in files:
                token = _upload_file(file_path=os.path.join(root, file), item_type=item_type, quiet=quiet)
                if token:
                    current_dict.files.append(token)

    return root_dict


def _upload_file(file_path: str, *, quiet: bool, item_type: str) -> Optional[str]:
    """Helper function to upload a single file.

    Args:
        full_path: path to the file to upload
        quiet: suppress verbose output
        item_type: Type of the item that is being uploaded.

    Returns:
        A str token of uploaded file if successful, otherwise None.
    """

    if not quiet:
        logger.info("Starting upload for file " + file_path)

    if not os.path.isfile(file_path):
        logger.warn("Skip uploading %s because it is not a file.", file_path)
        return None

    content_length = os.path.getsize(file_path)
    token = _upload_blob(file_path, item_type)
    if not quiet:
        logger.info("Upload successful: " + file_path + " (" + File.get_size(content_length) + ")")
    return token


def normalize_patterns(*, default: list[str], additional: Optional[Union[list[str], str]]) -> list[str]:
    """Merges additional patterns with the default, and normalize the dir pattern with wildcard."""

    def add_wildcard_to_dir(pattern: str) -> str:
        return pattern + "*" if pattern.endswith("/") else pattern

    if additional is None:
        additional = []
    elif isinstance(additional, str):
        additional = [additional]

    return [add_wildcard_to_dir(pattern) for pattern in default + additional]
