import logging
import os
import time
import zipfile
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union

import requests
from requests.exceptions import ConnectionError, Timeout
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
        files: Optional[List[str]] = None,
        directories: Optional[List["UploadDirectoryInfo"]] = None,
    ):
        self.name = name
        self.files = files if files is not None else []
        self.directories = directories if directories is not None else []

    def serialize(self) -> Dict:
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
        return "%.*f%s" % (precision, size, suffixes[suffix_index])


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
        except (ConnectionError, Timeout):
            logger.info(f"Network issue while checking uploaded size, retrying in {backoff_factor} seconds...")
            time.sleep(backoff_factor)
            backoff_factor = min(backoff_factor * 2, 60)
            retry_count += 1

    return 0  # Return 0 if all retries fail


def _upload_blob(file_path: str, model_type: str) -> str:
    """Uploads a file to a remote server as a blob and returns an upload token.

    Parameters
    ==========
    file_path: The path to the file to be uploaded.
    model_type : The type of the model associated with the file.
    """
    file_size = os.path.getsize(file_path)
    data = {
        "type": model_type,
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
    headers = {"Content-Type": "application/octet-stream", "Content-Range": f"bytes 0-{file_size - 1}/{file_size}"}

    retry_count = 0
    uploaded_bytes = 0
    backoff_factor = 1  # Initial backoff duration in seconds

    with open(file_path, "rb") as f, tqdm(total=file_size, desc="Uploading", unit="B", unit_scale=True) as pbar:
        while uploaded_bytes < file_size and retry_count < MAX_RETRIES:
            try:
                f.seek(uploaded_bytes)
                reader_wrapper = CallbackIOWrapper(pbar.update, f, "read")
                headers["Content-Range"] = f"bytes {uploaded_bytes}-{file_size - 1}/{file_size}"
                upload_response = requests.put(
                    session_uri, headers=headers, data=reader_wrapper, timeout=REQUEST_TIMEOUT
                )

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
    folder: str, model_type: str, quiet: bool = False  # noqa: FBT002, FBT001
) -> UploadDirectoryInfo:
    # Count the total number of files
    file_count = 0
    for _, _, files in os.walk(folder):
        file_count += len(files)

    if file_count > MAX_FILES_TO_UPLOAD:
        if not quiet:
            logger.info(f"More than {MAX_FILES_TO_UPLOAD} files detected, creating a zip archive...")

        with TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, TEMP_ARCHIVE_FILE)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for root, _, files in os.walk(folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder))

            tokens = [
                token
                for token in [_upload_file_or_folder(temp_dir, TEMP_ARCHIVE_FILE, model_type, quiet)]
                if token is not None
            ]
            return UploadDirectoryInfo(name="archive", files=tokens)

    root_dict = UploadDirectoryInfo(name="root")
    if os.path.isfile(folder):
        # Directly upload the file if the path is a file
        file_name = os.path.basename(folder)
        token = _upload_file_or_folder(os.path.dirname(folder), file_name, model_type, quiet)
        if token:
            root_dict.files.append(token)
    else:
        for root, _, files in os.walk(folder):
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
                token = _upload_file_or_folder(root, file, model_type, quiet)
                if token:
                    current_dict.files.append(token)

    return root_dict


def _upload_file_or_folder(
    parent_path: str,
    file_or_folder_name: str,
    model_type: str,
    quiet: bool = False,  # noqa: FBT002, FBT001
) -> Optional[str]:
    """
    Uploads a file or each file inside a folder individually from a specified path to a remote service.
    Parameters
    ==========
    parent_path: The parent directory path from where the file or folder is to be uploaded.
    file_or_folder_name: The name of the file or folder to be uploaded.
    dir_mode: The mode to handle directories. Accepts 'zip', 'tar', or other values for skipping.
    model_type: Type of the model that is being uploaded.
    quiet: suppress verbose output (default is False)
    :return: A token if the upload is successful, or None if the file is skipped or the upload fails.
    """
    full_path = os.path.join(parent_path, file_or_folder_name)
    if os.path.isfile(full_path):
        return _upload_file(file_or_folder_name, full_path, quiet, model_type)
    return None


def _upload_file(file_name: str, full_path: str, quiet: bool, model_type: str) -> Optional[str]:  # noqa: FBT001
    """Helper function to upload a single file
    Parameters
    ==========
    file_name: name of the file to upload
    full_path: path to the file to upload
    quiet: suppress verbose output
    model_type: Type of the model that is being uploaded.
    :return: None - upload unsuccessful; instance of UploadFile - upload successful
    """

    if not quiet:
        logger.info("Starting upload for file " + file_name)

    content_length = os.path.getsize(full_path)
    token = _upload_blob(full_path, model_type)
    if not quiet:
        logger.info("Upload successful: " + file_name + " (" + File.get_size(content_length) + ")")
    return token
