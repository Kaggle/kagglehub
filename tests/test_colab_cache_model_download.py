import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

import requests

import kagglehub
from kagglehub.clients import ColabClient
from kagglehub.colab_cache_resolver import COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME
from kagglehub.config import DISABLE_COLAB_CACHE_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .utils import create_test_server_colab

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/1"
LATEST_MODEL_VERSION = 2
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"
UNAVAILABLE_MODEL_HANDLE = "unavailable/model/handle/colab/1"


class ColabTBERuntimeHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(content_length))
        version = LATEST_MODEL_VERSION
        if "version" in request:
            version = request["version"]

        slug = f"{request['model']}/{request['framework']}/{request['variation']}/{version}"

        if self.path.endswith(ColabClient.IS_SUPPORTED_PATH):
            if request["owner"] == "unavailable":
                self.send_response(404)
                self.send_header("Content-type", "application/json")
                self.end_headers()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

        elif self.path.endswith(ColabClient.MOUNT_PATH):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            # Load the files
            cache_mount_folder = os.getenv(COLAB_CACHE_MOUNT_FOLDER_ENV_VAR_NAME)
            base_path = f"{cache_mount_folder}/{slug}"
            os.makedirs(base_path, exist_ok=True)

            Path(f"{base_path}/config.json").touch()

            if version == LATEST_MODEL_VERSION:
                # The latest version has an extra file.
                Path(f"{base_path}/model.keras").touch()

            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "slug": slug,
                        }
                    ),
                    "utf-8",
                )
            )
        else:
            self.send_response(404)
            self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))


class TestColabCacheModelDownload(BaseTestCase):
    def test_unversioned_model_download(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            model_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/2"))
            self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

    def test_versioned_model_download(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path)))

    def test_versioned_model_download_with_path(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            model_file_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_unversioned_model_download_with_path(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            model_file_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_versioned_model_download_with_missing_file_raises(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            with self.assertRaises(ValueError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, "missing.txt")

    def test_unversioned_model_download_with_missing_file_raises(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            with self.assertRaises(ValueError):
                kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "missing.txt")

    def test_colab_resolver_skipped_when_disable_colab_cache_env_var_name(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_COLAB_CACHE_ENV_VAR_NAME: "true"}):
            with create_test_server_colab(ColabTBERuntimeHandler):
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.model_download(VERSIONED_MODEL_HANDLE)

    def test_colab_resolver_skipped_when_model_not_present(self) -> None:
        with create_test_server_colab(ColabTBERuntimeHandler):
            # Assert that a ConnectionError is set (uses HTTP server which is not set)
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(UNAVAILABLE_MODEL_HANDLE)

    def test_versioned_model_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.model_download("bad handle")
