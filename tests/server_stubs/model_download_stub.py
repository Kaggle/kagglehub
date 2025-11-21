import hashlib
import os
from collections.abc import Generator
from typing import Any

from flask import Flask, Response, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.models.types.model_api_service import (
    ApiDownloadModelInstanceVersionRequest,
    ApiGetModelInstanceRequest,
    ApiListModelInstanceVersionFilesRequest,
    ApiModelInstance,
)

from kagglehub.enum import enum_to_str
from kagglehub.integrity import to_b64_digest
from tests.utils import get_test_file_path

app = Flask(__name__)

INVALID_ARCHIVE_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
ZIP_ARCHIVE_HANDLE = "testorg/testmodel/jax/zip/3"
TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE = "testorg/testmodel/jax/too-many-files/1"

# See https://cloud.google.com/storage/docs/xml-api/reference-headers#xgooghash
GCS_HASH_HEADER = "x-goog-hash"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/models.ModelApiService/GetModelInstance", methods=["POST"])
def model_get_instance() -> ResponseReturnValue:
    r = ApiGetModelInstanceRequest.from_dict(request.get_json())
    instance = ApiModelInstance()
    instance.framework = r.framework
    instance.slug = r.instance_slug
    instance.version_number = 3
    return instance.to_json(), 200


@app.route("/api/v1/models.ModelApiService/DownloadModelInstanceVersion", methods=["POST"])
def model_download_instance_version() -> ResponseReturnValue:
    r = ApiDownloadModelInstanceVersionRequest.from_dict(request.get_json())
    handle = f"{r.owner_slug}/{r.model_slug}/{enum_to_str(r.framework)}/{r.instance_slug}/{r.version_number}"

    if r.path:
        test_file_path = get_test_file_path(r.path)

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

    if handle == INVALID_ARCHIVE_HANDLE:
        return "bad archive", 200

    test_file_path = get_test_file_path("archive.tar.gz")
    content_type = "application/x-gzip"
    if handle == ZIP_ARCHIVE_HANDLE:
        test_file_path = get_test_file_path("archive.zip")
        content_type = "application/zip"

    with open(test_file_path, "rb") as f:
        content = f.read()
        file_hash = hashlib.md5()
        file_hash.update(content)
        resp = Response()
        resp.headers[GCS_HASH_HEADER] = f"md5={to_b64_digest(file_hash)}"
        resp.content_type = content_type
        resp.content_length = os.path.getsize(test_file_path)
        resp.data = content
        return resp, 200


@app.route("/api/v1/models.ModelApiService/ListModelInstanceVersionFiles", methods=["POST"])
def model_list_files() -> ResponseReturnValue:
    r = ApiListModelInstanceVersionFilesRequest.from_dict(request.get_json())
    handle = f"{r.owner_slug}/{r.model_slug}/{enum_to_str(r.framework)}/{r.instance_slug}/{r.version_number}"
    if handle in (INVALID_ARCHIVE_HANDLE, TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE):
        data = {
            "files": [{"name": f"{i}.txt"} for i in range(1, 51)],
            "nextPageToken": "more",
        }
    elif handle == ZIP_ARCHIVE_HANDLE:
        data = {
            "files": [{"name": f"model-{i}.txt"} for i in range(1, 27)],
            "nextPageToken": "more",
        }
    else:
        data = {
            "files": [
                {"name": "config.json"},
                {"name": "model.keras"},
            ],
            "nextPageToken": "",
        }
    return jsonify(data), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404
