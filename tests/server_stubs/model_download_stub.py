import hashlib
import os
from typing import Any, Generator

from flask import Flask, Response, jsonify
from flask.typing import ResponseReturnValue

from kagglehub.http_resolver import MODEL_INSTANCE_VERSION_FIELD
from kagglehub.integrity import to_b64_digest
from tests.utils import get_test_file_path

app = Flask(__name__)


# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/get", methods=["GET"])
def model_get_instance(org_slug: str, model_slug: str, framework: str, variation: str) -> ResponseReturnValue:
    data = {
        "message": f"Instance exists {org_slug}/{model_slug}/{framework}/{variation} !",
        MODEL_INSTANCE_VERSION_FIELD: 3,
    }
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/<version>/download", methods=["GET"])
def model_download_instance_version(
    org_slug: str, model_slug: str, framework: str, variation: str, version: int
) -> ResponseReturnValue:
    _ = f"{org_slug}/{model_slug}/{framework}/{variation}/{version}"
    test_file_path = get_test_file_path("archive.tar.gz")
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


@app.route(
    "/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/<version>/download/<path:subpath>", methods=["GET"]
)
def model_download_instance_version_path(
    org_slug: str, model_slug: str, framework: str, variation: str, version: int, subpath: str
) -> ResponseReturnValue:
    _ = f"{org_slug}/{model_slug}/{framework}/{variation}/{version}"
    test_file_path = get_test_file_path(subpath)

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


@app.route(
    "/api/v1/models/<org_slug>/<model_slug>/<framework>/bad-archive-variation/<version>/download", methods=["GET"]
)
def model_download_bad_archive(org_slug: str, model_slug: str, framework: str, version: int) -> ResponseReturnValue:
    _ = f"{org_slug}/{model_slug}/{framework}/bad-archive-variation/{version}"
    return "bad archive", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
