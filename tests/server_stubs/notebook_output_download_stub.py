from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from kagglehub.http_resolver import NOTEBOOK_CURRENT_VERSION_FIELD
from tests.utils import (
    AUTO_COMPRESSED_FILE_NAME,
    add_mock_gcs_route,
    get_gcs_redirect_response,
)

app = Flask(__name__)
add_mock_gcs_route(app)

TARGZ_ARCHIVE_HANDLE = "testuser/zip-notebook/versions/1"
# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"
LAST_MODIFIED = "Last-Modified"
LAST_MODIFIED_DATE = "Thu, 02 Mar 2020 02:17:12 GMT"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/kernels/pull", methods=["GET"])
def notebook_get() -> ResponseReturnValue:
    user_name = request.args.get("user_name")
    kernel_slug = request.args.get("kernel_slug")
    data = {"message": f"Notebook exists {user_name}/{kernel_slug} !", "metadata": {NOTEBOOK_CURRENT_VERSION_FIELD: 2}}
    return jsonify(data), 200


@app.route("/api/v1/kernels/output/download/<owner_slug>/<kernel_slug>", methods=["GET"])
def notebook_output_download(owner_slug: str, kernel_slug: str) -> ResponseReturnValue:
    handle = f"{owner_slug}/{kernel_slug}"

    # First, determine if we're fetching a file or the whole notebook output
    file_name_query_param = request.args.get("file_path")
    if file_name_query_param:
        # This mimics behavior for our file downloads, where users request a file, but
        # receive a zipped version of the file from GCS.
        test_file_name = (
            f"{AUTO_COMPRESSED_FILE_NAME}.zip"
            if file_name_query_param == AUTO_COMPRESSED_FILE_NAME
            else file_name_query_param
        )
    # Check a special case to handle tar.gz
    elif handle in TARGZ_ARCHIVE_HANDLE:
        test_file_name = "archive.tar.gz"
    else:
        test_file_name = "foo.txt.zip"

    return get_gcs_redirect_response(test_file_name)


@app.route("/api/v1/kernels/output/list/<owner_slug>/<kernel_slug>", methods=["GET"])
def notebook_list_files(owner_slug: str, kernel_slug: str) -> ResponseReturnValue:
    _ = f"{owner_slug}/{kernel_slug}"

    data = {"files": [{"url": "testUrl", "fileName": "foo.txt"}]}

    return jsonify(data), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some erorr response data", "error": str(e)}
    return jsonify(data), 404
