import os

import kagglehub
from kagglehub.cache import NOTEBOOKS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_notebook_handle
from tests.fixtures import BaseTestCase

from .server_stubs import notebook_output_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

INVALID_ARCHIVE_NOTEBOOK_OUTPUT_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_NOTEBOOK_OUTPUT_HANDLE = "khsamaha/simple-lightgbm-kaggle-sticker-sales-py/versions/2"
UNVERSIONED_NOTEBOOK_OUTPUT_HANDLE = "khsamaha/simple-lightgbm-kaggle-sticker-sales-py"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"


EXPECTED_NOTEBOOK_SUBDIR = os.path.join(
    NOTEBOOKS_CACHE_SUBFOLDER, "khsamaha", "simple-lightgbm-kaggle-sticker-sales-py", "output", "versions", "2"
)


class TestHttpNotebookOutputDownload(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _download_notebook_output_and_assert_downloaded(
        self,
        d: str,
        notebook_handle: str,
        expected_subdir_or_subpath: str,
        expected_files: list[str] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full notebook output and ensure all files are there.
        notebook_path = kagglehub.notebook_output_download(notebook_handle, **kwargs)
        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), notebook_path)

        if not expected_files:
            expected_files = ["foo.txt"]
        self.assertEqual(sorted(expected_files), sorted(os.listdir(notebook_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(parse_notebook_handle(notebook_handle))
        self.assertFalse(os.path.exists(archive_path))

    def _download_test_file_and_assert_downloaded(self, d: str, notebook_handle: str, **kwargs) -> None:  # noqa: ANN003
        notebook_path = kagglehub.notebook_output_download(notebook_handle, path=TEST_FILEPATH, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_NOTEBOOK_SUBDIR, TEST_FILEPATH), notebook_path)
        with open(notebook_path) as notebook_file:
            self.assertEqual(TEST_CONTENTS, notebook_file.read())

    def test_notebook_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.notebook_output_download(INVALID_ARCHIVE_NOTEBOOK_OUTPUT_HANDLE)

    def test_unversioned_notebook_output_download(self) -> None:
        with create_test_cache() as d:
            self._download_notebook_output_and_assert_downloaded(
                d, UNVERSIONED_NOTEBOOK_OUTPUT_HANDLE, EXPECTED_NOTEBOOK_SUBDIR
            )

    def test_versioned_notebook_output_download(self) -> None:
        with create_test_cache() as d:
            self._download_notebook_output_and_assert_downloaded(
                d, VERSIONED_NOTEBOOK_OUTPUT_HANDLE, EXPECTED_NOTEBOOK_SUBDIR
            )

    def test_versioned_dataset_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, VERSIONED_NOTEBOOK_OUTPUT_HANDLE)
