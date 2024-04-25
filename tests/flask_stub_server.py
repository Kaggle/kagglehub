import threading

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

app = Flask(__name__)

APACHE_LICENSE = "Apache 2.0"
ALLOWED_LICENSE_VALUES = APACHE_LICENSE


shared_data = {"files": [], "simulate_308": False, "blob_request_count": 0}
lock = threading.Lock()


def _increment_blob_request() -> None:
    lock.acquire()
    shared_data["blob_request_count"] += 1
    lock.release()


def _add_file(file: str) -> None:
    lock.acquire()
    shared_data["files"].append(file)
    lock.release()


def reset() -> None:
    lock.acquire()
    shared_data["files"] = []
    shared_data["blob_request_count"] = 0
    shared_data["simulate_308"] = 0
    lock.release()


def simulate_308(*, state: bool) -> None:
    lock.acquire()
    shared_data["simulate_308"] = state
    lock.release()


@app.route("/", methods=["HEAD"])
def head():  # noqa: ANN201
    return "", 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/get", methods=["GET"])
def model_get(org_slug: str, model_slug: str):  # noqa: ANN201
    data = {"message": f"Model exists {org_slug}/{model_slug} !"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/get", methods=["GET"])
def model_get_instance(org_slug: str, model_slug: str, framework: str, variation: str):  # noqa: ANN201
    data = {"message": f"Instance exists {org_slug}/{model_slug}/{framework}/{variation} !"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/<version>/get", methods=["GET"])
def model_get_instance_version(org_slug: str, model_slug: str, framework: str, variation: str, version: int):  # noqa: ANN201
    data = {"message": f"Instance exists {org_slug}/{model_slug}/{framework}/{variation}/{version} !"}
    return jsonify(data), 200


@app.errorhandler(404)
def error():  # noqa: ANN201
    data = {"message": "Some response data"}
    return jsonify(data), 404


@app.route("/api/v1/models/create/new", methods=["POST"])
def model_create():  # noqa: ANN201
    data = {"status": "success", "message": "Model created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/create/instance", methods=["POST"])
def model_instance_create_instance(org_slug: str, model_slug: str):  # noqa: ANN201
    post_data = request.get_json()
    # TODO: error here
    if post_data.get("licenseName", "") not in ALLOWED_LICENSE_VALUES:
        data = {"error": f"bad: {request.path}"}
        return jsonify(data), 200
    data = {"status": "success", "message": f"Model Instance {org_slug}/{model_slug} created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/models/<org_slug>/<model_slug>/<framework>/<variation>/create/version", methods=["POST"])
def model_instance_create_version(org_slug: str, model_slug: str, framework: str, variation: str):  # noqa: ANN201
    post_data = request.get_json()
    # TODO: error here
    if post_data.get("licenseName", "") not in ALLOWED_LICENSE_VALUES:
        data = {"error": f"bad: {request.path}"}
        return jsonify(data), 200
    data = {
        "status": "success",
        "message": f"Model Version {org_slug}/{model_slug}/{framework}/{variation} created successfully",
    }
    return jsonify(data), 200


@app.route("/api/v1/blobs/upload", methods=["POST"])
def blob_upload():  # noqa: ANN201
    if shared_data["simulate_308"] and shared_data["blob_request_count"] < 0:
        _increment_blob_request()
        return "", 308
    data = {
        "token": "dummy",
        "createUrl": "http://127.0.0.1:7777//upload/storage/v1/b/kaggle-models-data/o",
        "status": "success",
        "message": "Here is your token and Url",
    }
    post_data = request.get_json()
    _add_file(post_data["name"])
    _increment_blob_request()
    return jsonify(data), 200


@app.route("/upload/storage/v1/b/kaggle-models-data/o", methods=["PUT"])
def model_instance_create():  # noqa: ANN201
    data = {"status": "success", "message": "File uploaded"}
    return jsonify(data), 200


class ServerThread(threading.Thread):
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server("127.0.0.1", 7777, app)
        self.ctx = app.app_context()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()


def start_server() -> None:
    global server  # noqa: PLW0603
    server = ServerThread(app)
    server.start()


def stop_server() -> None:
    global server  # noqa: PLW0602
    server.shutdown()


if __name__ == "__main__":
    app.run("127.0.0.1", port=7777, debug=False, use_reloader=False)
