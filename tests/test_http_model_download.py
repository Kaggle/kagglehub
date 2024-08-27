import os
from typing import Optional

import requests

import kagglehub
from kagglehub.cache import MODELS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_model_handle
from tests.fixtures import BaseTestCase

from .server_stubs import model_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/3"
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"
TEST_CONTENTS = "{}"

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
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _download_model_and_assert_downloaded(
        self,
        d: str,
        model_handle: str,
        expected_subdir_or_subpath: str,
        expected_files: Optional[list[str]] = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full model and ensure all files are there.
        model_path = kagglehub.model_download(model_handle, **kwargs)
        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), model_path)
        if not expected_files:
            expected_files = ["config.json", "model.keras"]
        self.assertEqual(sorted(expected_files), sorted(os.listdir(model_path)))

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
            self._download_model_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_versioned_model_download(self) -> None:
        with create_test_cache() as d:
            self._download_model_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_model_archive_targz_download(self) -> None:
        with create_test_cache() as d:
            self._download_model_and_assert_downloaded(
                d,
                stub.TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE,
                f"{MODELS_CACHE_SUBFOLDER}/{stub.TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE}",
                expected_files=[f"{i}.txt" for i in range(1, 51)],
            )

    def test_model_archive_zip_download(self) -> None:
        with create_test_cache() as d:
            self._download_model_and_assert_downloaded(
                d,
                stub.ZIP_ARCHIVE_HANDLE,
                f"{MODELS_CACHE_SUBFOLDER}/{stub.ZIP_ARCHIVE_HANDLE}",
                expected_files=[f"model-{i}.txt" for i in range(1, 27)],
            )

    def test_versioned_model_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
            self._download_model_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_unversioned_model_full_download_with_file_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
            self._download_model_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

    def test_unversioned_model_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_model_and_assert_downloaded(
                d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
            )

    def test_versioned_model_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_model_and_assert_downloaded(
                d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
            )

    def test_versioned_model_full_download_with_file_already_cached_and_force_download(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
            self._download_model_and_assert_downloaded(
                d, VERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
            )

    def test_unversioned_model_full_download_with_file_already_cached_and_force_download(self) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
            self._download_model_and_assert_downloaded(
                d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
            )

    def test_versioned_model_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.model_download(stub.INVALID_ARCHIVE_HANDLE)

    def test_versioned_model_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE)

    def test_unversioned_model_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_with_path_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, VERSIONED_MODEL_HANDLE, force_download=True)

    def test_unversioned_model_download_with_path_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, force_download=True)

    def test_versioned_model_download_already_cached(self) -> None:
        with create_test_cache() as d:
            # Download from server.
            kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), model_path)

    def test_versioned_model_download_with_path_already_cached(self) -> None:
        with create_test_cache() as d:
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # No internet, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)

    def test_versioned_model_download_already_cached_with_force_download_explicit_false(self) -> None:
        with create_test_cache() as d:
            kagglehub.model_download(VERSIONED_MODEL_HANDLE)

            # Not force downloaded, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=False)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBDIR), model_path)

    def test_versioned_model_download_with_path_already_cached_with_force_download_explicit_false(self) -> None:
        with create_test_cache() as d:
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)

            # Not force downloaded, cache hit.
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH, force_download=False)

            self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)


class TestHttpNoInternet(BaseTestCase):
    def test_versioned_model_download_already_cached_with_force_download(self) -> None:
        with create_test_cache():
            server = serv.start_server(stub.app)
            kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            server.shutdown()

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=True)

    def test_versioned_model_download_with_path_already_cached_with_force_download(self) -> None:
        with create_test_cache():
            server = serv.start_server(stub.app)
            kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
            server.shutdown()

            # No internet should throw an error.
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH, force_download=True)
