import hashlib
import os

from flask import Flask, Response, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest, ApiDownloadDataFilesRequest

from kagglehub.integrity import to_b64_digest
from tests.utils import AUTO_COMPRESSED_FILE_NAME, add_mock_gcs_route, get_gcs_redirect_response, get_test_file_path

app = Flask(__name__)
add_mock_gcs_route(app)

TARGZ_ARCHIVE_HANDLE = "competition-targz"

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"
LAST_MODIFIED = "Last-Modified"
LAST_MODIFIED_DATE = "Thu, 02 Mar 2020 02:17:12 GMT"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/competitions.CompetitionApiService/DownloadDataFiles", methods=["POST"])
def competition_download() -> ResponseReturnValue:
    r = ApiDownloadDataFilesRequest.from_dict(request.get_json())
    handle = f"{r.competition_name}"

    test_file_path = get_test_file_path("foo.txt.zip")
    content_type = "application/zip"

    if handle in TARGZ_ARCHIVE_HANDLE:
        test_file_path = get_test_file_path("archive.tar.gz")
        content_type = "application/x-gzip"

    with open(test_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        resp = Response()
        resp.headers[GCS_HASH_HEADER] = f"md5={to_b64_digest(file_hash)}"
        resp.headers[LAST_MODIFIED] = LAST_MODIFIED_DATE
        resp.content_type = content_type
        resp.content_length = os.path.getsize(test_file_path)
        resp.data = content
        return resp, 200


@app.route("/api/v1/competitions.CompetitionApiService/DownloadDataFile", methods=["POST"])
def competition_download_file() -> ResponseReturnValue:
    r = ApiDownloadDataFileRequest.from_dict(request.get_json())

    # This mimics behavior for our file downloads, where users request a file, but
    # receive a zipped version of the file from GCS.
    test_file = f"{r.file_name}.zip" if r.file_name is AUTO_COMPRESSED_FILE_NAME else r.file_name
    return get_gcs_redirect_response(test_file)


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
