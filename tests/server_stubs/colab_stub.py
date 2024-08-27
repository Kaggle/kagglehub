import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from flask import Flask, jsonify, request
from flask.typing import ResponseReturnValue

from kagglehub.clients import (
    ColabClient,
)
from kagglehub.colab_cache_resolver import COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME

app = Flask(__name__)

LATEST_MODEL_VERSION = 2


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route(ColabClient.IS_SUPPORTED_PATH, methods=["POST"])
def models_is_supported() -> ResponseReturnValue:
    data = request.get_json()
    version = LATEST_MODEL_VERSION
    if "version" in data:
        version = data["version"]
    model = data["model"]
    framework = data["framework"]
    variation = data["variation"]
    slug = f"{model}/{framework}/{variation}/{version}"
    if data["owner"] == "unavailable":
        return slug, 400
    return "", 200


@app.route(ColabClient.MOUNT_PATH, methods=["POST"])
def models_mount() -> ResponseReturnValue:
    data = request.get_json()
    version = LATEST_MODEL_VERSION
    if "version" in data:
        version = data["version"]
    model = data["model"]
    framework = data["framework"]
    variation = data["variation"]
    slug = f"{model}/{framework}/{variation}/{version}"
    cache_mount_folder = os.getenv(COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
    base_path = f"{cache_mount_folder}/{slug}"
    os.makedirs(base_path, exist_ok=True)
    Path(f"{base_path}/config.json").touch()
    if version == LATEST_MODEL_VERSION:
        # The latest version has an extra file.
        Path(f"{base_path}/model.keras").touch()
    return jsonify({"slug": slug}), 200


@contextmanager
def create_env() -> Generator[Any, Any, Any]:
    with TemporaryDirectory() as cache_mount_folder:
        with mock.patch.dict(
            os.environ,
            {
                COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME: cache_mount_folder,
            },
        ):
            yield
