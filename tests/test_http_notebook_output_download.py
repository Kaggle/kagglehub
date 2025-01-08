import os
from typing import Optional

from tests.fixtures import BaseTestCase

import kagglehub
from kagglehub.handle import parse_notebook_handle
from .server_stubs import notebook_output_download_stub as stub
from .server_stubs import serv
from kagglehub.cache import NOTEBOOKS_CACHE_SUBFOLDER, get_cached_archive_path
from .utils import create_test_cache

INVALID_ARCHIVE_NOTEBOOK_OUTPUT_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_NOTEBOOK_OUTPUT_HANDLE = "jeward/testingNotebookOutput/versions/2"
UNVERSIONED_NOTEBOOK_OUTPUT_HANDLE = "khsamaha/simple-lightgbm-kaggle-sticker-sales-py"

EXPECTED_NOTEBOOK_SUBDIR = os.path.join(
    NOTEBOOKS_CACHE_SUBFOLDER, "khsamaha", "simple-lightgbm-kaggle-sticker-sales-py", "output"
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
        expected_files: Optional[list[str]] = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full notebook output and ensure all files are there.
        notebook_path = kagglehub.notebook_output_download(notebook_handle, **kwargs)
        print(f"Notebook path: {notebook_path}")
        print(f"Expected path: {os.path.join(d, expected_subdir_or_subpath)}")
        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), notebook_path)

        if not expected_files:
            expected_files = ["foo.txt"]
        self.assertEqual(sorted(expected_files), sorted(os.listdir(notebook_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(parse_notebook_handle(notebook_handle))
        self.assertFalse(os.path.exists(archive_path))

    def test_notebook_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.notebook_output_download(INVALID_ARCHIVE_NOTEBOOK_OUTPUT_HANDLE)

    def test_unversioned_notebook_output_download(self) -> None:
        with create_test_cache() as d:
            self._download_notebook_output_and_assert_downloaded(d, UNVERSIONED_NOTEBOOK_OUTPUT_HANDLE, EXPECTED_NOTEBOOK_SUBDIR)
