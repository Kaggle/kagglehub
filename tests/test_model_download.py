import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from unittest import mock
from urllib.parse import urlparse

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER
from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, KAGGLE_API_ENDPOINT_ENV_VAR_NAME

from .utils import get_test_file_path

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"


class FileHTTPHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
        if self.path.endswith(VERSIONED_MODEL_HANDLE + "/download/" + TEST_FILEPATH):
            test_file_path = get_test_file_path(TEST_FILEPATH)
            with open(test_file_path, "rb") as f:
                # Serve a single file
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", os.path.getsize(test_file_path))
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path.endswith(VERSIONED_MODEL_HANDLE + "/download"):
            test_file_path = get_test_file_path("archive.tar.gz")
            with open(test_file_path, "rb") as f:
                # Serve archive file
                self.send_response(200)
                self.send_header("Content-type", "application/x-gzip")
                self.send_header("Content-Length", os.path.getsize(test_file_path))
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path.endswith(INVALID_ARCHIVE_MODEL_HANDLE + "/download"):
            # Serve a bad archive file
            content = b"bad archive"
            self.send_response(200)
            self.send_header("Content-type", "application/x-gzip")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)  # bad archive
        else:
            # Unknown file path
            return self.send_response(404)


class TestModelDownload(unittest.TestCase):
    def test_unversioned_model_download(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)

    def test_versioned_model_download(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
                        self.assertEqual(
                            os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                            model_path,
                        )
                        self.assertEqual(["config.json", "model.keras"], os.listdir(model_path))
                    finally:
                        httpd.shutdown()

    def test_versioned_model_full_download_with_file_already_cached(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        # Download a single file
                        kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                        # Then download the full model and ensure all files are there.
                        model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

                        self.assertEqual(
                            os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                            model_path,
                        )
                        self.assertEqual(["config.json", "model.keras"], os.listdir(model_path))
                    finally:
                        httpd.shutdown()

    def test_versioned_model_download_bad_archive(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        with self.assertRaises(ValueError):
                            kagglehub.model_download(INVALID_ARCHIVE_MODEL_HANDLE)
                    finally:
                        httpd.shutdown()

    def test_versioned_model_download_with_path(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                        self.assertEqual(
                            os.path.join(
                                d,
                                MODELS_CACHE_SUBFOLDER,
                                "metaresearch",
                                "llama-2",
                                "pyTorch",
                                "13b",
                                "3",
                                TEST_FILEPATH,
                            ),
                            model_path,
                        )
                        with open(model_path) as model_file:
                            self.assertEqual("{}", model_file.readline())
                    finally:
                        httpd.shutdown()

    def test_versioned_model_download_already_cached(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                # Download from server.
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        kagglehub.model_download(VERSIONED_MODEL_HANDLE)
                    finally:
                        httpd.shutdown()

                # No internet, cache hit.
                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

                self.assertEqual(
                    os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                    model_path,
                )

    def test_versioned_model_download_with_path_already_cached(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                # Download from server.
                test_server_address = urlparse(os.getenv(KAGGLE_API_ENDPOINT_ENV_VAR_NAME))
                with HTTPServer((test_server_address.hostname, test_server_address.port), FileHTTPHandler) as httpd:
                    threading.Thread(target=httpd.serve_forever).start()

                    try:
                        kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                    finally:
                        httpd.shutdown()

                # No internet, cache hit.
                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

                self.assertEqual(
                    os.path.join(
                        d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3", TEST_FILEPATH
                    ),
                    model_path,
                )
