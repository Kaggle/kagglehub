import os
import unittest
from http.server import BaseHTTPRequestHandler

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER

from .utils import create_test_cache, create_test_http_server, get_test_file_path

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
        with create_test_cache() as d:
            with create_test_http_server(FileHTTPHandler):
                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
                self.assertEqual(
                    os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                    model_path,
                )
                self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

    def test_versioned_model_full_download_with_file_already_cached(self):
        with create_test_cache() as d:
            with create_test_http_server(FileHTTPHandler):
                # Download a single file
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                # Then download the full model and ensure all files are there.
                model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

                self.assertEqual(
                    os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                    model_path,
                )
                self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

    def test_versioned_model_download_bad_archive(self):
        with create_test_cache():
            with create_test_http_server(FileHTTPHandler):
                with self.assertRaises(ValueError):
                    kagglehub.model_download(INVALID_ARCHIVE_MODEL_HANDLE)

    def test_versioned_model_download_with_path(self):
        with create_test_cache() as d:
            with create_test_http_server(FileHTTPHandler):
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

    def test_versioned_model_download_already_cached(self):
        with create_test_cache() as d:
            # Download from server.
            with create_test_http_server(FileHTTPHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            self.assertEqual(
                os.path.join(d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3"),
                model_path,
            )

    def test_versioned_model_download_with_path_already_cached(self):
        with create_test_cache() as d:
            with create_test_http_server(FileHTTPHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(
                os.path.join(
                    d, MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3", TEST_FILEPATH
                ),
                model_path,
            )
