import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler

import requests

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_model_handle
from kagglehub.http_resolver import MODEL_INSTANCE_VERSION_FIELD
from kagglehub.integrity import GCS_HASH_HEADER, to_b64_digest
from tests.fixtures import BaseTestCase

from .utils import create_test_cache, create_test_http_server, get_test_file_path

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"
TEST_CONTENTS = "{}"


class KaggleAPIHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith(VERSIONED_MODEL_HANDLE + "/download/" + TEST_FILEPATH):
            test_file_path = get_test_file_path(TEST_FILEPATH)
            with open(test_file_path, "rb") as f:
                # Serve a single file
                self.send_response(200)
                self.send_header("Content-type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(test_file_path)))

                content = f.read()
                file_hash = hashlib.md5()
                file_hash.update(content)
                self.send_header(GCS_HASH_HEADER, f"md5={to_b64_digest(file_hash)}")
                self.end_headers()
                self.wfile.write(content)
        elif self.path.endswith(VERSIONED_MODEL_HANDLE + "/download"):
            test_file_path = get_test_file_path("archive.tar.gz")
            with open(test_file_path, "rb") as f:
                # Serve archive file
                self.send_response(200)
                self.send_header("Content-type", "application/x-gzip")
                self.send_header("Content-Length", str(os.path.getsize(test_file_path)))
                content = f.read()
                file_hash = hashlib.md5()
                file_hash.update(content)
                self.send_header(GCS_HASH_HEADER, f"md5={to_b64_digest(file_hash)}")
                self.end_headers()
                self.wfile.write(content)
        elif self.path.endswith(INVALID_ARCHIVE_MODEL_HANDLE + "/download"):
            # Serve a bad archive file
            content = b"bad archive"
            self.send_response(200)
            self.send_header("Content-type", "application/x-gzip")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)  # bad archive
        elif self.path.endswith(UNVERSIONED_MODEL_HANDLE + "/get"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({MODEL_INSTANCE_VERSION_FIELD: 3}), "utf-8"))
        else:
            self.send_response(404)
            self.wfile.write(bytes(f"Unhandled path: {self.path}", "utf-8"))


EXPECTED_MODEL_SUBDIR = os.path.join(MODELS_CACHE_SUBFOLDER, "metaresearch", "llama-2", "pyTorch", "13b", "3")
EXPECTED_MODEL_SUBPATH = os.path.join(
    MODELS_CACHE_SUBFOLDER,
    "metaresearch",
    "llama-2",
    "pyTorch",
    "13b",
    "3",
    TEST_FILEPATH,
)


# Test cases for the ModelHttpResolver.
class TestHttpModelDownload(BaseTestCase):
    def _download_model_and_assert_downloaded(
        self, d: str, model_handle: str, expected_subdir_or_subpath: str, **kwargs  # noqa: ANN003
    ) -> None:
        # Download the full model and ensure all files are there.
        model_path = kagglehub.model_download(model_handle, **kwargs)
        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), model_path)
        self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

        # Assert that the archive file has been deleted.
        archive_path = get_cached_archive_path(parse_model_handle(model_handle))
        self.assertFalse(os.path.exists(archive_path))

    def _download_test_file_and_assert_downloaded(self, d: str, model_handle: str, **kwargs) -> None:  # noqa: ANN003
        model_path = kagglehub.model_download(model_handle, path=TEST_FILEPATH, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)
        with open(model_path) as model_file:
            self.assertEqual(TEST_CONTENTS, model_file.readline())

    def test_unversioned_model_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_model_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_versioned_model_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_model_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_versioned_model_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                # Download a single file first
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                self._download_model_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_unversioned_model_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                # Download a single file first
                kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                self._download_model_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_unversioned_model_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_model_and_assert_downloaded(
                    d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
                )

    def test_versioned_model_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_model_and_assert_downloaded(
                    d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
                )

    def test_versioned_model_full_download_with_file_already_cached_and_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                # Download a single file first
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                self._download_model_and_assert_downloaded(
                    d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
                )

    def test_unversioned_model_full_download_with_file_already_cached_and_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                # Download a single file first
                kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
                self._download_model_and_assert_downloaded(
                    d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
                )

    def test_versioned_model_download_bad_archive(self) -> None:
        with create_test_cache():
            with create_test_http_server(KaggleAPIHandler):
                with self.assertRaises(ValueError):
                    kagglehub.model_download(INVALID_ARCHIVE_MODEL_HANDLE)

    def test_versioned_model_download_with_path(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_test_file_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE)

    def test_unversioned_model_download_with_path(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_with_path_with_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_test_file_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, force_download=True)

    def test_unversioned_model_download_with_path_with_force_download(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, force_download=True)

    def test_versioned_model_download_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download from server.
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), model_path)

    def test_versioned_model_download_with_path_already_cached(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)

    def test_versioned_model_download_already_cached_with_force_download(self) -> None:
        with create_test_cache():
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=True)

    def test_versioned_model_download_with_path_already_cached_with_force_download(self) -> None:
        with create_test_cache():
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH, force_download=True)

    def test_versioned_model_download_already_cached_with_force_download_explicit_false(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # Not force downloaded, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=False)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), model_path)

    def test_versioned_model_download_with_path_already_cached_with_force_download_explicit_false(self) -> None:
        with create_test_cache() as d:
            with create_test_http_server(KaggleAPIHandler):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # Not force downloaded, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH, force_download=False)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)
