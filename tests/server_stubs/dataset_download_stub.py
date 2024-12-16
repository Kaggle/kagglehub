from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from kagglehub.http_resolver import DATASET_CURRENT_VERSION_FIELD
from tests.utils import AUTO_COMPRESSED_FILE_NAME, add_mock_gcs_route, get_gcs_redirect_response

app = Flask(__name__)
add_mock_gcs_route(app)

TARGZ_ARCHIVE_HANDLE = "testuser/zip-dataset/versions/1"


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

    return get_gcs_redirect_response(test_file_name)


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
