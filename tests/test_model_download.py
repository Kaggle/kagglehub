import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import threading

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_path
from kagglehub.config import CACHE_FOLDER_ENV_VAR_NAME, KAGGLE_API_ENDPOINT_ENV_VAR_NAME
from kagglehub.handle import parse_model_handle
from .utils import get_test_file_path

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"


class FileHTTPHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)

    def do_GET(self):
        if self.path.endswith(VERSIONED_MODEL_HANDLE + "/download/foo.txt"):
            test_file_path = get_test_file_path("foo.txt")
            with open(test_file_path, 'rb') as f:
                # Serve a single file
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", os.path.getsize(test_file_path))
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path.endswith(VERSIONED_MODEL_HANDLE + "/download"):
            test_file_path = get_test_file_path("archive.tar.gz")
            with open(test_file_path, 'rb') as f:
                # Serve archive file
                self.send_response(200)
                self.send_header("Content-type", "application/x-gzip")
                self.send_header("Content-Length", os.path.getsize(test_file_path))
                self.end_headers()
                self.wfile.write(f.read())
        elif self.path.endswith(INVALID_ARCHIVE_MODEL_HANDLE + "/download"):
                # Serve a bad archive file
                content = "bad archive".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/x-gzip")
                self.send_header("Content-Length", len(content))
                self.end_headers()
                self.wfile.write(content) # bad archive
        
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
                        # TODO: assert on file inside the archive
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
                        model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path="foo.txt")
                        self.assertEqual(
                            os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3", "foo.txt"),
                            model_path,
                        )
                        with open(model_path) as model_file:
                            self.assertEqual("bar", model_file.readline())
                    finally:
                        httpd.shutdown()

    def test_versioned_model_download_already_cached(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                cache_path = get_cached_path(parse_model_handle(VERSIONED_MODEL_HANDLE))
                os.makedirs(cache_path)

                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

                self.assertEqual(
                    os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                    model_path,
                )

    def test_versioned_model_download_with_path_already_cached(self):
        with TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {CACHE_FOLDER_ENV_VAR_NAME: d}):
                cache_path = get_cached_path(parse_model_handle(VERSIONED_MODEL_HANDLE))
                os.makedirs(cache_path)
                Path(os.path.join(cache_path, "foo.txt")).touch()  # Create file

                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path="foo.txt")

                self.assertEqual(
                    os.path.join(
                        d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3", "foo.txt"
                    ),
                    model_path,
                )
