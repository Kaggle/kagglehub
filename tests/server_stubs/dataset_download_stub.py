import threading
from dataclasses import dataclass

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.datasets.types.dataset_api_service import ApiDataset, ApiDownloadDatasetRequest, ApiGetDatasetRequest

from tests.utils import AUTO_COMPRESSED_FILE_NAME, add_mock_gcs_route, get_gcs_redirect_response

app = Flask(__name__)
add_mock_gcs_route(app)

TARGZ_ARCHIVE_HANDLE = "testuser/zip-dataset/versions/1"


@dataclass
class SharedData:
    last_download_user_agent = ""


shared_data: SharedData = SharedData()
lock = threading.Lock()


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/datasets.DatasetApiService/GetDataset", methods=["POST"])
def dataset_get() -> ResponseReturnValue:
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
    lock.acquire()
    shared_data.last_download_user_agent = request.headers.get("User-Agent", "")
    lock.release()

    r = ApiDownloadDatasetRequest.from_dict(request.get_json())

    handle = f"{r.owner_slug}/{r.dataset_slug}"

    # First, determine if we're fetching a file or the whole dataset
    if r.file_name:
        # This mimics behavior for our file downloads, where users request a file, but
        # receive a zipped version of the file from GCS.
        test_file_name = f"{AUTO_COMPRESSED_FILE_NAME}.zip" if r.file_name == AUTO_COMPRESSED_FILE_NAME else r.file_name
    # Check a special case to handle tar.gz
    elif handle in TARGZ_ARCHIVE_HANDLE:
        test_file_name = "archive.tar.gz"
    else:
        test_file_name = "foo.txt.zip"

    return get_gcs_redirect_response(test_file_name)


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
