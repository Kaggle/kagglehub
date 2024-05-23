import hashlib
import os
from typing import Any, Generator

from flask import Flask, jsonify, Response
from flask.typing import ResponseReturnValue

from kagglehub.integrity import to_b64_digest
from tests.utils import get_test_file_path

app = Flask(__name__)

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"

@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/datsets/view/<owner_slug>/<dataset_slug>", methods=["GET"])
def dataset_get(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    data = {"message": f"Dataset exists {owner_slug}/{dataset_slug} !"}
    return jsonify(data), 200


@app.route("/api/v1/datasets/download/<owner_slug>/<dataset_slug>", methods=["GET"])
def dataset_download(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    _ = f"{owner_slug}/{dataset_slug}"
    test_file_path = get_test_file_path("foo.txt")
    with open(test_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        resp = Response()
        resp.headers[GCS_HASH_HEADER] = f"md5={to_b64_digest(file_hash)}"
        resp.content_type = "application/x-gzip"
        resp.content_length = os.path.getsize(test_file_path)
        resp.data = content
        return resp, 200
    
@app.route("/api/v1/datasets/download/<owner_slug>/<dataset_slug>/<file_name>", methods=["GET"])
def dataset_download_file(owner_slug: str, dataset_slug: str, file_name: str) -> ResponseReturnValue:
    _ = f"{owner_slug}/{dataset_slug}"
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
                headers={GCS_HASH_HEADER: f"md5={to_b64_digest(file_hash)}", "Content-Length": str(len(content))},
            ),
            200,
        )
    
@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
