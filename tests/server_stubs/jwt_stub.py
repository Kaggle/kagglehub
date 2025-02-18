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
LATEST_KERNEL_VERSION = 2

PACKAGE_INIT_PY_TEXT = """
__package_version__ = '0.1.0'

import kagglehub

module = __import__(__name__)
__all__ = kagglehub.packages._finalize_package_import(module)
"""

PACKAGE_REQUIREMENTS_YAML_TEXT = """
format_version: 0.1.0
datasources: []
"""

PACKAGE_FOO_PY_TEXT = """
def foo():
    return "kaggle"

__all__ = ["foo"]
"""

PACKAGE_BAR_PY_TEXT = """
import kagglehub

def bar():
    path = kagglehub.get_package_asset_path('asset.txt')
    with open(path) as f:
        return f.read()

__all__ = ["bar"]
"""

PACKAGE_ASSET_TEXT = "abcd"


@app.route("/", methods=["HEAD"])
def head() -> ResponseReturnValue:
    return "", 200


@app.errorhandler(404)
def error(e: Exception):  # noqa: ANN201
    data = {"code": "404", "error": str(e)}
    return jsonify(data), 200


@app.route("/kaggle-jwt-handler/AttachDatasourceUsingJwtRequest", methods=["POST"])
def attach_datasource_using_jwt_request() -> ResponseReturnValue:
    cache_mount_folder = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
    if not cache_mount_folder:
        msg = f"Missing envvar '{KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME}'"
        raise ValueError(msg)

    data = request.get_json()
    if "modelRef" in data:
        model_ref = data["modelRef"]
        version_number = LATEST_MODEL_VERSION
        if "VersionNumber" in model_ref:
            version_number = model_ref["VersionNumber"]
        mount_slug = f"{model_ref['ModelSlug']}/{model_ref['Framework']}/{model_ref['InstanceSlug']}/{version_number}"
        # Load the files
        base_path = Path(cache_mount_folder) / mount_slug
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / "config.json").touch()
        if version_number == LATEST_MODEL_VERSION:
            # The latest version has an extra file.
            (base_path / "model.keras").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
                "versionNumber": version_number,
            },
        }
        return jsonify(data), 200
    elif "datasetRef" in data:
        dataset_ref = data["datasetRef"]
        version_number = LATEST_DATASET_VERSION
        if "VersionNumber" in dataset_ref:
            version_number = dataset_ref["VersionNumber"]
        mount_slug = f"{dataset_ref['DatasetSlug']}"
        # Load the files
        base_path = Path(cache_mount_folder) / mount_slug
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / "foo.txt").touch()
        if version_number == LATEST_DATASET_VERSION:
            # The latest version has an extra file.
            (base_path / "bar.csv").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
                "versionNumber": version_number,
            },
        }
        return jsonify(data), 200
    elif "competitionRef" in data:
        competition_ref = data["competitionRef"]
        mount_slug = f"{competition_ref['CompetitionSlug']}"
        # Load the files
        base_path = Path(cache_mount_folder) / mount_slug
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / "foo.txt").touch()
        (base_path / "bar.csv").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
            },
        }
        return jsonify(data), 200
    elif "kernelRef" in data:
        kernel_ref = data["kernelRef"]
        version_number = LATEST_KERNEL_VERSION
        if "VersionNumber" in kernel_ref:
            version_number = kernel_ref["VersionNumber"]
        mount_slug = f"{kernel_ref['KernelSlug']}"
        # Load the files
        base_path = Path(cache_mount_folder) / mount_slug
        base_path.mkdir(parents=True, exist_ok=True)
        latest_version = version_number == LATEST_KERNEL_VERSION
        if mount_slug == "test-package":
            _write_package_files(base_path, latest_version)
        else:
            (base_path / "foo.txt").touch()
            if version_number == LATEST_KERNEL_VERSION:
                # The latest version has an extra file.
                (base_path / "bar.csv").touch()
        data = {
            "wasSuccessful": True,
            "result": {
                "mountSlug": mount_slug,
                "versionNumber": version_number,
            },
        }
        return jsonify(data), 200
    else:
        return jsonify(data), 500


def _write_package_files(base_path: Path, latest_version: bool) -> None:  # noqa: FBT001
    package_path = base_path / "package"
    package_path.mkdir(parents=True, exist_ok=True)

    (package_path / "__init__.py").write_text(PACKAGE_INIT_PY_TEXT)
    (package_path / "kagglehub_requirements.yaml").write_text(PACKAGE_REQUIREMENTS_YAML_TEXT)
    (package_path / "foo.py").write_text(PACKAGE_FOO_PY_TEXT)

    if latest_version:
        (package_path / "bar.py").write_text(PACKAGE_BAR_PY_TEXT)
        assets_path = package_path / "assets"
        assets_path.mkdir(parents=True, exist_ok=True)
        (package_path / "assets" / "asset.txt").write_text(PACKAGE_ASSET_TEXT)


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
