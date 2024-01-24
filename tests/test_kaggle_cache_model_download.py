import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.kaggle_cache_resolver import ATTACH_DATASOURCE_REQUEST_NAME, KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .utils import create_test_jwt_http_server

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/1"
LATEST_MODEL_VERSION = 2
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"


class KaggleJwtHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.endswith(ATTACH_DATASOURCE_REQUEST_NAME):
            content_length = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(content_length))
            model_ref = request["modelRef"]
            version_number = LATEST_MODEL_VERSION
            if "VersionNumber" in model_ref:
                version_number = model_ref["VersionNumber"]

            mount_slug = (
                f"{model_ref['ModelSlug']}/{model_ref['Framework']}/{model_ref['InstanceSlug']}/{version_number}"
            )

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            # Load the files
            cache_mount_folder = os.getenv(KAGGLE_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
            base_path = f"{cache_mount_folder}/{mount_slug}"
            os.makedirs(base_path, exist_ok=True)

            Path(f"{base_path}/config.json").touch()

            if version_number == LATEST_MODEL_VERSION:
                # The latest version has an extra file.
                Path(f"{base_path}/model.keras").touch()

            # Return the response
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "wasSuccessful": True,
                            "result": {
                                "mountSlug": mount_slug,
                            },
                        }
                    ),
                    "utf-8",
                )
            )
        else:
            self.send_response(404)
            self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))


# Test cases for the ModelKaggleCacheResolver.
class TestKaggleCacheModelDownload(BaseTestCase):
    def test_unversioned_model_download(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/2"))
            self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

    def test_versioned_model_download(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path)))

    def test_versioned_model_download_with_path(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_file_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_unversioned_model_download_with_path(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_file_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_versioned_model_download_with_missing_file_raises(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            with self.assertRaises(ValueError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, "missing.txt")

    def test_unversioned_model_download_with_missing_file_raises(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            with self.assertRaises(ValueError):
                kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "missing.txt")

    def test_kaggle_resolver_skipped(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_KAGGLE_CACHE_ENV_VAR_NAME: "true"}):
            with create_test_jwt_http_server(KaggleJwtHandler):
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.model_download(VERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.model_download("bad handle")

    def test_versioned_model_download_with_force_download(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            model_path_forced = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=True)

            # Using force_download shouldn't change the expected output of model_download.
            self.assertTrue(model_path_forced.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path_forced)))
            self.assertEqual(model_path, model_path_forced)

    def test_versioned_model_download_with_force_download_explicitly_false(self) -> None:
        with create_test_jwt_http_server(KaggleJwtHandler):
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=False)
            self.assertTrue(model_path.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path)))
