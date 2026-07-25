import threading
from dataclasses import dataclass, field

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.datasets.databundles.types.databundle_api_types import ApiDirectory, ApiDirectoryContent, ApiFile
from kagglesdk.datasets.types.dataset_api_service import (
    ApiDataset,
    ApiDownloadDatasetRequest,
    ApiGetDatasetRequest,
    ApiListTreeDatasetFilesRequest,
)

from tests.utils import AUTO_COMPRESSED_FILE_NAME, add_mock_gcs_route, get_gcs_redirect_response

app = Flask(__name__)
add_mock_gcs_route(app)

TARGZ_ARCHIVE_HANDLE = "testuser/zip-dataset/versions/1"


@dataclass
class SharedData:
    last_download_user_agent = ""
    download_requests = 0
    tree_requests = 0
    get_dataset_requests = 0
    fail_once_paths: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.last_download_user_agent = ""
        self.download_requests = 0
        self.tree_requests = 0
        self.get_dataset_requests = 0
        self.fail_once_paths = set()


shared_data: SharedData = SharedData()
lock = threading.Lock()


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/datasets.DatasetApiService/GetDataset", methods=["POST"])
def dataset_get() -> ResponseReturnValue:
    with lock:
        shared_data.get_dataset_requests += 1
    r = ApiGetDatasetRequest.from_dict(request.get_json())
    dataset = ApiDataset()
    dataset.owner_name = r.owner_slug
    dataset.title = r.dataset_slug
    dataset.current_version_number = 2
    return dataset.to_json(), 200


# For Datasets, downloads of the archive and individual files happen at the same route, controlled
# by a file_name query param
@app.route("/api/v1/datasets.DatasetApiService/DownloadDataset", methods=["POST"])
def dataset_download() -> ResponseReturnValue:
    with lock:
        shared_data.last_download_user_agent = request.headers.get("User-Agent", "")
        shared_data.download_requests += 1

    r = ApiDownloadDatasetRequest.from_dict(request.get_json())

    handle = f"{r.owner_slug}/{r.dataset_slug}"

    # First, determine if we're fetching a file, a folder, or the whole dataset
    folder_paths = {"empty", "folder", "partial", "malicious", "loop", "missing"}
    if r.file_name in folder_paths:
        return jsonify({"message": "No exact file at this path"}), 404

    if r.file_name:
        with lock:
            if r.file_name in shared_data.fail_once_paths:
                shared_data.fail_once_paths.remove(r.file_name)
                return jsonify({"message": "Injected transient failure"}), 500

        # This mimics behavior for our file downloads, where users request a file, but
        # receive a zipped version of the file from GCS.
        if r.file_name == AUTO_COMPRESSED_FILE_NAME:
            test_file_name = f"{AUTO_COMPRESSED_FILE_NAME}.zip"
        elif r.file_name == "folder/page2.csv":
            test_file_name = AUTO_COMPRESSED_FILE_NAME
        elif r.file_name.startswith(("folder/", "partial/", "loop/")):
            test_file_name = "foo.txt"
        else:
            test_file_name = r.file_name
    # Check a special case to handle tar.gz
    elif handle in TARGZ_ARCHIVE_HANDLE:
        test_file_name = "archive.tar.gz"
    else:
        test_file_name = "foo.txt.zip"

    return get_gcs_redirect_response(test_file_name)


def _directory(relative_url: str) -> ApiDirectory:
    value = ApiDirectory()
    value.name = relative_url.rsplit("/", 1)[-1]
    value.relative_url = relative_url
    return value


def _file(relative_url: str) -> ApiFile:
    value = ApiFile()
    value.name = relative_url.rsplit("/", 1)[-1]
    value.relative_url = relative_url
    return value


@app.route("/api/v1/datasets.DatasetApiService/ListTreeDatasetFiles", methods=["POST"])
def dataset_list_tree() -> ResponseReturnValue:
    with lock:
        shared_data.tree_requests += 1

    r = ApiListTreeDatasetFilesRequest.from_dict(request.get_json())
    response = ApiDirectoryContent()

    if r.path == "folder" and not r.page_token:
        response.directories = [_directory("folder/nested")]
        response.files = [_file("folder/root.txt")]
        response.next_page_token = "folder-page-2"
    elif r.path == "folder" and r.page_token == "folder-page-2":
        response.files = [_file("folder/page2.csv")]
    elif r.path == "folder/nested" and not r.page_token:
        response.files = [_file("folder/nested/child.txt")]
        response.next_page_token = "nested-page-2"
    elif r.path == "folder/nested" and r.page_token == "nested-page-2":
        response.files = [_file("folder/nested/child2.txt")]
    elif r.path == "partial":
        response.files = [_file("partial/first.txt"), _file("partial/second.txt")]
    elif r.path == "malicious":
        response.files = [_file("malicious/../../escape.txt")]
    elif r.path == "loop" and not r.page_token:
        response.files = [_file("loop/file.txt")]
        response.next_page_token = "repeated-token"
    elif r.path == "loop" and r.page_token == "repeated-token":
        response.next_page_token = "repeated-token"

    return response.to_json(), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
