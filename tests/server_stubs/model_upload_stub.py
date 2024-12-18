import datetime
import threading
from dataclasses import dataclass, field
from datetime import timedelta

import jwt
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


@app.route("/api/v1/models/signing/token", methods=["POST"])
def model_signing_token() -> ResponseReturnValue:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    private_key = b"-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAwhvqCC+37A+UXgcvDl+7nbVjDI3QErdZBkI1VypVBMkKKWHM\nNLMdHk0bIKL+1aDYTRRsCKBy9ZmSSX1pwQlO/3+gRs/MWG27gdRNtf57uLk1+lQI\n6hBDozuyBR0YayQDIx6VsmpBn3Y8LS13p4pTBvirlsdX+jXrbOEaQphn0OdQo0WD\noOwwsPCNCKoIMbUOtUCowvjesFXlWkwG1zeMzlD1aDDS478PDZdckPjT96ICzqe4\nO1Ok6fRGnor2UTmuPy0f1tI0F7Ol5DHAD6pZbkhB70aTBuWDGLDR0iLenzyQecmD\n4aU19r1XC9AHsVbQzxHrP8FveZGlV/nJOBJwFwIDAQABAoIBAFCVFBA39yvJv/dV\nFiTqe1HahnckvFe4w/2EKO65xTfKWiyZzBOotBLrQbLH1/FJ5+H/82WVboQlMATQ\nSsH3olMRYbFj/NpNG8WnJGfEcQpb4Vu93UGGZP3z/1B+Jq/78E15Gf5KfFm91PeQ\nY5crJpLDU0CyGwTls4ms3aD98kNXuxhCGVbje5lCARizNKfm/+2qsnTYfKnAzN+n\nnm0WCjcHmvGYO8kGHWbFWMWvIlkoZ5YubSX2raNeg+YdMJUHz2ej1ocfW0A8/tmL\nwtFoBSuBe1Z2ykhX4t6mRHp0airhyc+MO0bIlW61vU/cPGPos16PoS7/V08S7ZED\nX64rkyECgYEA4iqeJZqny/PjOcYRuVOHBU9nEbsr2VJIf34/I9hta/mRq8hPxOdD\n/7ES/ZTZynTMnOdKht19Fi73Sf28NYE83y5WjGJV/JNj5uq2mLR7t2R0ZV8uK8tU\n4RR6b2bHBbhVLXZ9gqWtu9bWtsxWOkG1bs0iONgD3k5oZCXp+IWuklECgYEA27bA\n7UW+iBeB/2z4x1p/0wY+whBOtIUiZy6YCAOv/HtqppsUJM+W9GeaiMpPHlwDUWxr\n4xr6GbJSHrspkMtkX5bL9e7+9zBguqG5SiQVIzuues9Jio3ZHG1N2aNrr87+wMiB\nxX6Cyi0x1asmsmIBO7MdP/tSNB2ebr8qM6/6mecCgYBA82ZJfFm1+8uEuvo6E9/R\nyZTbBbq5BaVmX9Y4MB50hM6t26/050mi87J1err1Jofgg5fmlVMn/MLtz92uK/hU\nS9V1KYRyLc3h8gQQZLym1UWMG0KCNzmgDiZ/Oa/sV5y2mrG+xF/ZcwBkrNgSkO5O\n7MBoPLkXrcLTCARiZ9nTkQKBgQCsaBGnnkzOObQWnIny1L7s9j+UxHseCEJguR0v\nXMVh1+5uYc5CvGp1yj5nDGldJ1KrN+rIwMh0FYt+9dq99fwDTi8qAqoridi9Wl4t\nIXc8uH5HfBT3FivBtLucBjJgOIuK90ttj8JNp30tbynkXCcfk4NmS23L21oRCQyy\nlmqNDQKBgQDRvzEB26isJBr7/fwS0QbuIlgzEZ9T3ZkrGTFQNfUJZWcUllYI0ptv\ny7ShHOqyvjsC3LPrKGyEjeufaM5J8EFrqwtx6UB/tkGJ2bmd1YwOWFHvfHgHCZLP\n34ZNURCvxRV9ZojS1zmDRBJrSo7+/K0t28hXbiaTOjJA18XAyyWmGg==\n-----END RSA PRIVATE KEY-----\n"
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
        private_key,
        algorithm="RS256",
    )
    data = {"id_token": token}
    return jsonify(data), 200
