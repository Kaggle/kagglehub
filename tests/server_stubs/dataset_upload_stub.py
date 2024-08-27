import threading
from dataclasses import dataclass, field

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from ..utils import resolve_endpoint

app = Flask(__name__)


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


@app.route("/api/v1/datsets/view/<owner_slug>/<dataset_slug>", methods=["GET"])
def dataset_get(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    data = {"message": f"Dataset exists {owner_slug}/{dataset_slug} !"}
    return jsonify(data), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404


@app.route("/api/v1/datasets/create/new", methods=["POST"])
def dataset_create() -> ResponseReturnValue:
    data = {"status": "success", "message": "Dataset created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/datasets/create/version/<owner_slug>/<dataset_slug>", methods=["POST"])
def dataset_create_version(owner_slug: str, dataset_slug: str) -> ResponseReturnValue:
    data = {
        "status": "success",
        "message": f"Dataset Version {owner_slug}/{dataset_slug} created successfully",
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
        "createUrl": f"http://{address}:{port}/upload/storage/v1/b/kaggle-datasets-data/o",
        "status": "success",
        "message": "Here is your token and Url",
    }
    post_data = request.get_json()
    _add_file(post_data["name"])
    _increment_blob_request()
    return jsonify(data), 200


@app.route("/upload/storage/v1/b/kaggle-datasets-data/o", methods=["PUT"])
def dataset_up_create() -> ResponseReturnValue:
    data = {"status": "success", "message": "File uploaded"}
    return jsonify(data), 200
