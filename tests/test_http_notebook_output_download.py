import os
from typing import Optional
from unittest import mock

import requests

import kagglehub
from kagglehub.cache import CODE_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_model_handle
from tests.fixtures import BaseTestCase

from .server_stubs import notebook_output_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

EXPECTED_NOTEBOOK_OUTPUT_SUBDIR = os.path.join(CODE_CACHE_SUBFOLDER, stub.TEST_HANDLE, "output")
EXPECTED_NOTEBOOK_OUTPUT_SUBPATH = os.path.join(
    CODE_CACHE_SUBFOLDER,
    stub.TEST_HANDLE,
    "output",
    stub.TEST_FILE,
)


def test_download(mocker, tmpdir) -> None:
    mocker.patch("kagglehub.code.registry.notebook_output_resolver", return_value=tmpdir)
    result = kagglehub.notebook_output_download("some/notebook", force_download=True)
    assert result == tmpdir


# Test cases for the NotebookOutputHttpResolver.
# class TestHttpNotebookOutputDownload(BaseTestCase):
#     # @classmethod
#     # def setUpClass(cls):
#     #     cls.server = serv.start_server(stub.app)

#     # @classmethod
#     # def tearDownClass(cls):
#     #     cls.server.shutdown()
#     def test_download(self, mocker, tmpdir) -> None:
#         mocker.patch("kagglehub.code.registry.notebook_output_resolver", return_value=tmpdir)
#         result = kagglehub.notebook_output_download("some/notebook", force_download=True)
#         self.assertEqual(result, tmpdir)


#     def _download_notebook_output_and_assert_downloaded(
#         self,
#         d: str,
#         code_handle: str,
#         expected_subdir_or_subpath: str,
#         expected_file: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         # Download the full model and ensure all files are there.
#         model_path = kagglehub.notebook_output_download(code_handle, **kwargs)
#         self.assertEqual(os.path.join(d, expected_subdir_or_subpath), model_path)
#         if not expected_file:
#             expected_file = "submission.csv"
#         self.assertEqual(expected_file, os.listdir(model_path))

#         # Assert that the archive file has been deleted.
#         archive_path = get_cached_archive_path(parse_model_handle(code_handle))
#         self.assertFalse(os.path.exists(archive_path))

#     def _download_test_file_and_assert_downloaded(self, d: str, model_handle: str, **kwargs) -> None:
#         model_path = kagglehub.model_download(model_handle, path=TEST_FILEPATH, **kwargs)
#         self.assertEqual(os.path.join(d, EXPECTED_MODEL_SUBPATH), model_path)
#         with open(model_path) as model_file:
#             self.assertEqual(TEST_CONTENTS, model_file.readline())

#     def test_model_archive_targz_download(self, tmpdir) -> None:
#         with create_test_cache() as d:
#             self._download_model_and_assert_downloaded(
#                 d,
#                 stub.TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE,
#                 f"{CODE_CACHE_SUBFOLDER}/{stub.TOO_MANY_FILES_FOR_PARALLEL_DOWNLOAD_HANDLE}",
#                 expected_files=[f"{i}.txt" for i in range(1, 51)],
#             )

#     def test_download_with_file_already_cached(self) -> None:
#         with create_test_cache() as d:
#             # Download a single file first
#             kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
#             self._download_model_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR)

#     def test_download_with_force_download(self) -> None:
#         with create_test_cache() as d:
#             self._download_model_and_assert_downloaded(
#                 d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
#             )

#     def test_unversioned_model_full_download_with_file_already_cached_and_force_download(self) -> None:
#         with create_test_cache() as d:
#             # Download a single file first
#             kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
#             self._download_model_and_assert_downloaded(
#                 d, UNVERSIONED_MODEL_HANDLE, EXPECTED_MODEL_SUBDIR, force_download=True
#             )

#     def test_unversioned_model_download_with_path(self) -> None:
#         with create_test_cache() as d:
#             self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE)

#     def test_unversioned_model_download_with_path_with_force_download(self) -> None:
#         with create_test_cache() as d:
#             self._download_test_file_and_assert_downloaded(d, UNVERSIONED_MODEL_HANDLE, force_download=True)


# class TestHttpNoInternet(BaseTestCase):
#     def test_versioned_model_download_already_cached_with_force_download(self) -> None:
#         with create_test_cache():
#             server = serv.start_server(stub.app)
#             kagglehub.model_download(VERSIONED_MODEL_HANDLE)
#             server.shutdown()

#             # No internet should throw an error.
#             with self.assertRaises(requests.exceptions.ConnectionError):
#                 kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=True)

#     def test_versioned_model_download_with_path_already_cached_with_force_download(self) -> None:
#         with create_test_cache():
#             server = serv.start_server(stub.app)
#             kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH)
#             server.shutdown()

#             # No internet should throw an error.
#             with self.assertRaises(requests.exceptions.ConnectionError):
#                 kagglehub.model_download(VERSIONED_MODEL_HANDLE, path=TEST_FILEPATH, force_download=True)
