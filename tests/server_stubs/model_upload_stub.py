import threading
from dataclasses import dataclass, field

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from ..utils import resolve_endpoint

app = Flask(__name__)

APACHE_LICENSE = "Apache 2.0"
ALLOWED_LICENSE_VALUES = APACHE_LICENSE


@dataclass
class SharedData:
    files: list[str] = field(default_factory=list)
    simulate_308: bool = False
    blob_request_count: int = 0


shared_data: SharedData = SharedData()
lock = threading.Lock()


def _increment_blob_request() -> None:
    lock.acquire()
    shared_data.blob_request_count += 1
    lock.release()


def _add_file(file: str) -> None:
    lock.acquire()
    shared_data.files.append(file)
    lock.release()


def reset() -> None:
    lock.acquire()
    shared_data.files = []
    shared_data.blob_request_count = 0
    shared_data.simulate_308 = False
    lock.release()


def simulate_308(*, state: bool) -> None:
    lock.acquire()
    shared_data.simulate_308 = state
    lock.release()


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/get", methods=["GET"])
def model_get(org_slug: str, model_slug: str) -> ResponseReturnValue:
    data = {"message": f"Model exists {org_slug}/{model_slug} !"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/get", methods=["GET"])
def model_get_instance(org_slug: str, model_slug: str, framework: str, variation: str) -> ResponseReturnValue:
    data = {"message": f"Instance exists {org_slug}/{model_slug}/{framework}/{variation} !"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/<version>/get", methods=["GET"])
def model_get_instance_version(
    org_slug: str, model_slug: str, framework: str, variation: str, version: int
) -> ResponseReturnValue:
    data = {"message": f"Instance exists {org_slug}/{model_slug}/{framework}/{variation}/{version} !"}
    return jsonify(data), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404


@app.route("/api/v1/models/create/new", methods=["POST"])
def model_create() -> ResponseReturnValue:
    data = {"status": "success", "message": "Model created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/create/instance", methods=["POST"])
def model_instance_create_instance(org_slug: str, model_slug: str) -> ResponseReturnValue:
    post_data = request.get_json()
    if post_data.get("licenseName", "") not in ALLOWED_LICENSE_VALUES:
        data = {"error": f"bad: {request.path}"}
        return jsonify(data), 200
    data = {"status": "success", "message": f"Model Instance {org_slug}/{model_slug} created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/create/version", methods=["POST"])
def model_instance_create_version(
    org_slug: str, model_slug: str, framework: str, variation: str
) -> ResponseReturnValue:
    post_data = request.get_json()
    if post_data.get("licenseName", "") not in ALLOWED_LICENSE_VALUES:
        data = {"error": f"bad: {request.path}"}
        return jsonify(data), 200
    data = {
        "status": "success",
        "message": f"Model Version {org_slug}/{model_slug}/{framework}/{variation} created successfully",
    }
    return jsonify(data), 200


@app.route("/api/v1/blobs/upload", methods=["POST"])
def blob_upload() -> ResponseReturnValue:
    if shared_data.simulate_308 and shared_data.blob_request_count < 0:
        _increment_blob_request()
        return "", 308
    address, port = resolve_endpoint()
    data = {
        "token": "dummy",
        "createUrl": f"http://{address}:{port}/upload/storage/v1/b/kaggle-models-data/o",
        "status": "success",
        "message": "Here is your token and Url",
    }
    post_data = request.get_json()
    _add_file(post_data["name"])
    _increment_blob_request()
    return jsonify(data), 200


@app.route("/upload/storage/v1/b/kaggle-models-data/o", methods=["PUT"])
def model_instance_create() -> ResponseReturnValue:
    data = {"status": "success", "message": "File uploaded"}
    return jsonify(data), 200
