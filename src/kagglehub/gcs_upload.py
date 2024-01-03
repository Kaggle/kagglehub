import logging
import os
import zipfile
from datetime import datetime
from tempfile import TemporaryDirectory

import requests
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

from kagglehub.clients import KaggleApiV1Client
from kagglehub.exceptions import BackendError

logger = logging.getLogger(__name__)

MAX_FILES_TO_UPLOAD = 50
TEMP_ARCHIVE_FILE = "archive.zip"


def parse_datetime_string(string: str):
    time_formats = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%fZ"]
    for t in time_formats:
        try:
            return datetime.strptime(string[:26], t).replace(microsecond=0)  # noqa: DTZ007
        except:  # noqa: E722, S110
            pass
    return string


class File(object):  # noqa: UP004
    def __init__(self, init_dict):
        parsed_dict = {k: parse_datetime_string(v) for k, v in init_dict.items()}
        self.__dict__.update(parsed_dict)
        self.size = File.get_size(self.totalBytes)

    def __repr__(self):
        return self.ref

    @staticmethod
    def get_size(size, precision=0):
        suffixes = ["B", "KB", "MB", "GB", "TB"]
        suffix_index = 0
        while size >= 1024 and suffix_index < 4:  # noqa: PLR2004
            suffix_index += 1
            size /= 1024.0
        return "%.*f%s" % (precision, size, suffixes[suffix_index])


def _upload_blob(file_path: str, model_type: str):
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

    headers = {"Content-Type": "application/octet-stream"}

    # TODO(312511716): add resumable upload
    with open(file_path, "rb") as f:
        with tqdm(total=file_size, desc="Uploading", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            reader_wrapper = CallbackIOWrapper(pbar.update, f, "read")
            gcs_response = requests.put(response["createUrl"], data=reader_wrapper, headers=headers, timeout=600)
            if gcs_response.status_code not in [200, 201]:
                upload_fail_message = (
                    f"Upload failed with status code: {gcs_response.status_code}, Response: {gcs_response.text}"
                )
                raise BackendError(upload_fail_message)

    return response["token"]


def upload_files(folder: str, model_type: str, quiet: bool = False):  # noqa: FBT002, FBT001
    """upload files in a folder. Zips the files if there are more than 50.
    Parameters
    ==========
    folder: the folder to upload from
    quiet: suppress verbose output (default is False)
    model_type: Type of the model that is being uploaded.
    """

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

            # Upload the zip file
            return [_upload_file_or_folder(temp_dir, TEMP_ARCHIVE_FILE, model_type, quiet)]

    tokens = []
    for file_name in os.listdir(folder):
        tokens.append(_upload_file_or_folder(folder, file_name, model_type, quiet))

    return tokens


def _upload_file_or_folder(
    parent_path: str, file_or_folder_name: str, model_type: str, quiet: bool = False  # noqa: FBT002, FBT001
):
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

    elif os.path.isdir(full_path):
        for filename in os.listdir(full_path):
            file_path = os.path.join(full_path, filename)
            if os.path.isfile(file_path):
                _upload_file(filename, file_path, quiet, model_type)
            elif not quiet:
                logger.info(f"Skipping non-file item in directory: {filename}")
    elif not quiet:
        logger.info("Skipping: " + file_or_folder_name)
    return None


def _upload_file(file_name: str, full_path: str, quiet: bool, model_type: str):  # noqa: FBT001
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
