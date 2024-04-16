import logging
import os
import shutil
import threading
import time
import zipfile
from datetime import datetime
from multiprocessing import Manager, Pool, Queue
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union

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


def zip_file(args: Tuple[Path, Path, Queue, Path]) -> None:
    file_path, zip_path, update_queue, source_path_obj = args
    arcname = file_path.relative_to(source_path_obj)
    size = file_path.stat().st_size
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_STORED, allowZip64=True) as zipf:
        zipf.write(file_path, arcname)
    update_queue.put(size)


def zip_files(source_path_obj: Path, zip_path: Path, update_queue: Queue) -> None:
    files = [file for file in source_path_obj.rglob("*") if file.is_file()]
    args = [(file, zip_path, update_queue, source_path_obj) for file in files]

    with Pool() as pool:
        pool.map(zip_file, args)


def manage_progress(update_queue: Queue, pbar: tqdm) -> None:
    while True:
        size = update_queue.get()
        if size is None:
            break
        pbar.update(size)
        update_queue.task_done()


def upload_files(source_path: str, model_type: str) -> List[str]:
    source_path_obj = Path(source_path)
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        total_size = 0

        if source_path_obj.is_dir():
            for file_path in source_path_obj.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        elif source_path_obj.is_file():
            total_size = source_path_obj.stat().st_size
        else:
            path_error_message = "The source path does not point to a valid file or directory."
            raise ValueError(path_error_message)

        with Manager() as manager:  # noqa: F841
            update_queue = Queue()
            with tqdm(total=total_size, desc="Zipping", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                progress_thread = threading.Thread(target=manage_progress, args=(update_queue, pbar))
                progress_thread.start()

                if source_path_obj.is_dir():
                    zip_path = temp_dir_path / "archive.zip"
                    zip_files(source_path_obj, zip_path, update_queue)
                    upload_path = str(zip_path)
                elif source_path_obj.is_file():
                    temp_file_path = temp_dir_path / source_path_obj.name
                    shutil.copy(source_path_obj, temp_file_path)
                    update_queue.put(temp_file_path.stat().st_size)
                    upload_path = str(temp_file_path)

                update_queue.put(None)  # Signal to stop the progress thread
                progress_thread.join()

        return [token for token in [_upload_blob(upload_path, model_type)] if token]
