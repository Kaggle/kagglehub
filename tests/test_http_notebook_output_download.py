import os
from tempfile import TemporaryDirectory

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

    def test_versioned_notebook_output_download_with_output_dir(self) -> None:
        with TemporaryDirectory() as dest:
            notebook_path = kagglehub.notebook_output_download(VERSIONED_NOTEBOOK_OUTPUT_HANDLE, output_dir=dest)
            self.assertEqual(dest, notebook_path)
            self.assertEqual([".complete", "foo.txt"], sorted(os.listdir(dest)))

    def test_versioned_notebook_output_download_with_path_and_output_dir(self) -> None:
        with TemporaryDirectory() as dest:
            notebook_path = kagglehub.notebook_output_download(
                VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                path=TEST_FILEPATH,
                output_dir=dest,
            )
            self.assertEqual(os.path.join(dest, TEST_FILEPATH), notebook_path)
            with open(notebook_path) as notebook_file:
                self.assertEqual(TEST_CONTENTS, notebook_file.read())

    def test_versioned_notebook_output_download_with_path_and_output_dir_existing_file_fails(self) -> None:
        with TemporaryDirectory() as dest:
            dest_file = os.path.join(dest, TEST_FILEPATH)
            with open(dest_file, "w") as output_file:
                output_file.write("old")
            with self.assertRaises(FileExistsError):
                kagglehub.notebook_output_download(
                    VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                    path=TEST_FILEPATH,
                    output_dir=dest,
                )

    def test_versioned_notebook_output_download_with_output_dir_existing_dir_fails(self) -> None:
        with TemporaryDirectory() as dest:
            with open(os.path.join(dest, "old.txt"), "w") as output_file:
                output_file.write("old")
            with self.assertRaises(FileExistsError):
                kagglehub.notebook_output_download(
                    VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                    output_dir=dest,
                )

    def test_versioned_notebook_output_download_with_output_dir_overwrite(self) -> None:
        with TemporaryDirectory() as dest:
            # Download it and ensure completion marker is set.
            kagglehub.notebook_output_download(
                VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                output_dir=dest,
            )
            # Add a random file to ensure the directory is overriden.
            with open(os.path.join(dest, "old.txt"), "w") as output_file:
                output_file.write("old")
            notebook_path = kagglehub.notebook_output_download(
                VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                output_dir=dest,
                force_download=True,
            )
            self.assertEqual(dest, notebook_path)
            self.assertEqual([".complete", "foo.txt"], sorted(os.listdir(dest)))

    def test_versioned_notebook_output_download_with_path_and_output_dir_overwrite(self) -> None:
        with TemporaryDirectory() as dest:
            # Download it and ensure completion marker is set.
            kagglehub.notebook_output_download(
                VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                path=TEST_FILEPATH,
                output_dir=dest,
            )
            # Update the file to ensure overwrite works.
            dest_file = os.path.join(dest, TEST_FILEPATH)
            with open(dest_file, "w") as output_file:
                output_file.write("old")
            notebook_path = kagglehub.notebook_output_download(
                VERSIONED_NOTEBOOK_OUTPUT_HANDLE,
                path=TEST_FILEPATH,
                output_dir=dest,
                force_download=True,
            )
            self.assertEqual(dest_file, notebook_path)
            with open(notebook_path) as notebook_file:
                self.assertEqual(TEST_CONTENTS, notebook_file.read())
