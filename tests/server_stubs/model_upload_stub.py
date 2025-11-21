import datetime
import threading
from dataclasses import dataclass, field
from datetime import timedelta

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue
from kagglesdk.models.types.model_api_service import (
    ApiCreateModelInstanceRequest,
    ApiCreateModelResponse,
    ApiGetModelRequest,
    ApiModel,
)

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


@app.route("/api/v1/models.ModelApiService/GetModel", methods=["POST"])
def model_get() -> ResponseReturnValue:
    r = ApiGetModelRequest.from_dict(request.get_json())
    model = ApiModel()
    model.author = r.owner_slug
    model.slug = r.model_slug

    response = ApiCreateModelResponse()
    return response.to_json(), 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"message": "Some response data", "error": str(e)}
    return jsonify(data), 404


@app.route("/api/v1/models.ModelApiService/CreateModel", methods=["POST"])
def model_create() -> ResponseReturnValue:
    data = {"status": "success", "message": "Model created successfully"}
    return jsonify(data), 200


@app.route("/api/v1/models.ModelApiService/CreateModelInstance", methods=["POST"])
def model_instance_create_instance() -> ResponseReturnValue:
    r = ApiCreateModelInstanceRequest.from_dict(request.get_json())
    response = ApiCreateModelResponse()
    if r.body.license_name not in ALLOWED_LICENSE_VALUES:
        response.error = f"bad: {request.path}"
    if r.body.instance_slug == "new-version":
        response.error = "Already exists"
        response.error_code = 409  # Conflict
    return response.to_json(), 200


@app.route("/api/v1/models.ModelApiService/CreateModelInstanceVersion", methods=["POST"])
def model_instance_create_version() -> ResponseReturnValue:
    return ApiCreateModelResponse().to_json(), 200


@app.route("/api/v1/blobs.BlobApiService/StartBlobUpload", methods=["POST"])
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


@app.route("/api/v1/models.ModelApiService/CreateModelSigningToken", methods=["POST"])
def model_signing_token() -> ResponseReturnValue:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    # Generate the private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    # Serialize the private key to PEM format
    key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    token = jwt.encode(
        {
            "iss": "https://www.kaggle.com/api/v1/models/signing",
            "sub": "418540",
            "aud": "sigstore",
            "exp": int((now + timedelta(minutes=10)).timestamp()),
            "nbf": int(now.timestamp()),
            "iat": int(now.timestamp()),
            "email": "foo@gmail.com",
            "email_verified": True,
        },
        key,
        algorithm="RS256",
    )
    data = {"id_token": token}
    return jsonify(data), 200
