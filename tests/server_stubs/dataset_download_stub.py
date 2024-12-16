import hashlib
import mimetypes
import os
from collections.abc import Generator
from typing import Any

from flask import Flask, Response, jsonify, request
from flask.typing import ResponseReturnValue

from kagglehub.http_resolver import DATASET_CURRENT_VERSION_FIELD
from kagglehub.integrity import to_b64_digest
from tests.utils import MOCK_GCS_BUCKET_BASE_PATH, get_mocked_gcs_signed_url, get_test_file_path

app = Flask(__name__)

TARGZ_ARCHIVE_HANDLE = "testuser/zip-dataset/versions/1"
AUTO_COMPRESSED_FILE_NAME = "dummy.csv"

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"
LOCATION_HEADER = "Location"
CONTENT_LENGTH_HEADER = "Content-Length"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/datasets/view/<owner_slug>/<dataset_slug>", methods=["GET"])
def dataset_get(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    data = {
        "message": f"Dataset exists {owner_slug}/{dataset_slug} !",
        DATASET_CURRENT_VERSION_FIELD: 2,
    }
    return jsonify(data), 200


# For Datasets, downloads of the archive and individual files happen at the same route, controlled
# by a file_name query param
@app.route("/api/v1/datasets/download/<owner_slug>/<dataset_slug>", methods=["GET"])
def dataset_download(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    handle = f"{owner_slug}/{dataset_slug}"

    # First, determine if we're fetching a file or the whole dataset
    file_name_query_param = request.args.get("file_name")
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

    # All downloads, regardless of archive or file, happen via GCS signed URLs. We mock the 302 and handle
    # the redirect not only to be thorough--without this, the response.url in download_file (clients.py)
    # will not pick up on followed redirect URL being different from the originally requested URL.
    return (
        Response(
            headers={
                LOCATION_HEADER: get_mocked_gcs_signed_url(os.path.basename(test_file_name)),
                CONTENT_LENGTH_HEADER: 0,
            }
        ),
        302,
    )


# Route to handle the mocked GCS redirects
@app.route(f"{MOCK_GCS_BUCKET_BASE_PATH}/<file_name>", methods=["GET"])
def handle_mock_gcs_redirect(file_name: str) -> ResponseReturnValue:
    test_file_path = get_test_file_path(file_name)

    def generate_file_content() -> Generator[bytes, Any, None]:
        with open(test_file_path, "rb") as f:
            while True:
                chunk = f.read(4096)  # Read file in chunks
                if not chunk:
                    break
                yield chunk

    with open(test_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        return (
            Response(
                generate_file_content(),
                headers={
                    GCS_HASH_HEADER: f"md5={to_b64_digest(file_hash)}",
                    "Content-Length": os.path.getsize(test_file_path),
                    "Content-Type": mimetypes.guess_type(test_file_path)[0] or "application/octet-stream",
                },
            ),
            200,
        )


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
