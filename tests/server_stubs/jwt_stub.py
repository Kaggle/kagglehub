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
    KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME,
    KAGGLE_JWT_TOKEN_ENV_VAR_NAME,
)
from kagglehub.env import KAGGLE_NOTEBOOK_ENV_VAR_NAME
from kagglehub.kaggle_cache_resolver import KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME

app = Flask(__name__)

LATEST_MODEL_VERSION = 2
LATEST_DATASET_VERSION = 2


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/kaggle-jwt-handler/AttachDatasourceUsingJwtRequest", methods=["POST"])
def attach_datasource_using_jwt_request() -> ResponseReturnValue:
    data = request.get_json()
    if "modelRef" in data:
        model_ref = data["modelRef"]
        version_number = LATEST_MODEL_VERSION
        if "VersionNumber" in model_ref:
            version_number = model_ref["VersionNumber"]
        mount_slug = f"{model_ref['ModelSlug']}/{model_ref['Framework']}/{model_ref['InstanceSlug']}/{version_number}"
        # # Load the files
        cache_mount_folder = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
        base_path = f"{cache_mount_folder}/{mount_slug}"
        os.makedirs(base_path, exist_ok=True)
        Path(f"{base_path}/config.json").touch()
        if version_number == LATEST_MODEL_VERSION:
            # The latest version has an extra file.
            Path(f"{base_path}/model.keras").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
            },
        }
        return jsonify(data), 200
    elif "datasetRef" in data:
        dataset_ref = data["datasetRef"]
        version_number = LATEST_DATASET_VERSION
        if "VersionNumber" in dataset_ref:
            version_number = dataset_ref["VersionNumber"]
        mount_slug = f"{dataset_ref['DatasetSlug']}"
        # # Load the files
        cache_mount_folder = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
        base_path = f"{cache_mount_folder}/{mount_slug}"
        os.makedirs(base_path, exist_ok=True)
        Path(f"{base_path}/foo.txt").touch()
        if version_number == LATEST_DATASET_VERSION:
            # The latest version has an extra file.
            Path(f"{base_path}/bar.csv").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
            },
        }
        return jsonify(data), 200
    else:
        return jsonify(data), 500


@contextmanager
def create_env() -> Generator[Any, Any, Any]:
    with TemporaryDirectory() as cache_mount_folder:
        with mock.patch.dict(
            os.environ,
            {
                KAGGLE_NOTEBOOK_ENV_VAR_NAME: "Interactive",
                KAGGLE_JWT_TOKEN_ENV_VAR_NAME: "foo jwt token",
                KAGGLE_DATA_PROXY_TOKEN_ENV_VAR_NAME: "foo proxy token",
                KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME: cache_mount_folder,
            },
        ):
            yield
