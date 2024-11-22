import hashlib

from flask import Flask, Response, jsonify
from flask.typing import ResponseReturnValue

from kagglehub.integrity import to_b64_digest

app = Flask(__name__)

TEST_HANDLE = "alexisbcook/titanic-tutorial"
TEST_FILE = "submission.csv"
TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE = "some-owner/notebook-with-too-many-output-files"

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route(f"/api/v1/kernels/{TEST_HANDLE}/output/download", methods=["GET"])
def code_download_notebook_output() -> ResponseReturnValue:
    content = str.encode("PassengerId,Survived\n892,0\n")
    file_hash = hashlib.md5()
    file_hash.update(content)
    resp = Response()
    resp.headers[GCS_HASH_HEADER] = f"md5={to_b64_digest(file_hash)}"
    resp.content_type = "application/x-gzip"
    resp.content_length = len(content)
    resp.data = content
    return resp, 200


@app.route(f"/api/v1/kernels/{TEST_HANDLE}/output/download/{TEST_FILE}", methods=["GET"])
def code_download_notebook_output_path() -> ResponseReturnValue:
    content = str.encode("PassengerId,Survived\n892,0\n")
    file_hash = hashlib.md5()
    file_hash.update(content)
    resp = Response()
    resp.headers[GCS_HASH_HEADER] = f"md5={to_b64_digest(file_hash)}"
    resp.content_type = "application/x-gzip"
    resp.content_length = len(content)
    resp.data = content
    return resp, 200


@app.route("/api/v1/kernels/<owner_slug>/<notebook_slug>/list", methods=["GET"])
def code_list_notebook_output_files(owner_slug: str, notebook_slug: str) -> ResponseReturnValue:
    handle = f"{owner_slug}/{notebook_slug}"
    if handle == TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE:
        data = {
            "files": [{"name": f"{i}.txt"} for i in range(1, 51)],
            "nextPageToken": "more",
        }
    elif handle == TEST_HANDLE:
        data = {
            "files": [
                {"name": TEST_FILE},
            ],
            "nextPageToken": "",
        }
    else:
        data = {}
    return jsonify(data), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
