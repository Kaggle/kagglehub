import os
from tempfile import TemporaryDirectory

import kagglehub
from kagglehub.cache import DATASETS_CACHE_SUBFOLDER, Cache, get_cached_archive_path
from kagglehub.exceptions import KaggleApiHTTPError
from kagglehub.handle import parse_dataset_handle
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_download_stub as stub
from .server_stubs import serv
from .utils import AUTO_COMPRESSED_FILE_NAME, create_test_cache, parameterized

INVALID_ARCHIVE_DATASET_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata/versions/2"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"
AUTO_COMPRESSED_CONTENTS = """"shape","degrees","sides","color","date"
"square",360,4,"blue","2024-12-17"
"circle",360,,"red","2023-08-01"
"triangle",180,3,"green","2022-01-05"
"""

EXPECTED_DATASET_SUBDIR = os.path.join(
    DATASETS_CACHE_SUBFOLDER, "sarahjeffreson", "featured-spotify-artiststracks-with-metadata", "versions", "2"
)
EXPECTED_DATASET_SUBPATH = os.path.join(
    DATASETS_CACHE_SUBFOLDER,
    "sarahjeffreson",
    "featured-spotify-artiststracks-with-metadata",
    "versions",
    "2",
)


class TestHttpDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self) -> None:
        super().setUp()
        stub.shared_data.reset()

    def _download_dataset_and_assert_downloaded(
        self,
        d: str,
        dataset_handle: str,
        expected_subdir_or_subpath: str,
        expected_files: list[str] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full datasets and ensure all files are there.
        dataset_path = kagglehub.dataset_download(dataset_handle, **kwargs)

        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), dataset_path)

        if not expected_files:
            expected_files = ["foo.txt"]
        self.assertEqual(sorted(expected_files), sorted(os.listdir(dataset_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(parse_dataset_handle(dataset_handle))
        self.assertFalse(os.path.exists(archive_path))

    def _download_test_file_and_assert_downloaded(self, d: str, dataset_handle: str, **kwargs) -> None:  # noqa: ANN003
        dataset_path = kagglehub.dataset_download(dataset_handle, path=TEST_FILEPATH, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH, TEST_FILEPATH), dataset_path)
        with open(dataset_path) as dataset_file:
            self.assertEqual(TEST_CONTENTS, dataset_file.read())

    def _download_test_file_and_assert_downloaded_auto_compressed(
        self,
        d: str,
        dataset_handle: str,
        **kwargs,  # noqa: ANN003
    ) -> None:
        dataset_path = kagglehub.dataset_download(dataset_handle, path=AUTO_COMPRESSED_FILE_NAME, **kwargs)
        self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH, AUTO_COMPRESSED_FILE_NAME), dataset_path)
        with open(dataset_path) as dataset_file:
            self.assertEqual(AUTO_COMPRESSED_CONTENTS, dataset_file.read())

    def test_unversioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    def test_versioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, VERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    def test_versioned_dataset_targz_archive_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d,
                stub.TARGZ_ARCHIVE_HANDLE,
                f"{DATASETS_CACHE_SUBFOLDER}/{stub.TARGZ_ARCHIVE_HANDLE}",
                expected_files=[f"{i}.txt" for i in range(1, 51)],
            )

    def test_versioned_dataset_download_bad_archive(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(INVALID_ARCHIVE_DATASET_HANDLE)

    def test_versioned_dataset_download_with_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded(d, VERSIONED_DATASET_HANDLE)

    def test_exact_file_download_does_not_list_tree(self) -> None:
        with create_test_cache():
            kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path=TEST_FILEPATH)
        self.assertEqual(0, stub.shared_data.tree_requests)

    def test_folder_download_recurses_and_paginates(self) -> None:
        with create_test_cache() as d:
            folder_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="folder")

            self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH, "folder"), folder_path)
            self.assertEqual(
                ["nested/child.txt", "nested/child2.txt", "page2.csv", "root.txt"],
                sorted(
                    os.path.relpath(os.path.join(root, file), folder_path)
                    for root, _, files in os.walk(folder_path)
                    for file in files
                ),
            )
            self.assertEqual(4, stub.shared_data.tree_requests)

    def test_folder_download_is_reused_from_cache(self) -> None:
        with create_test_cache():
            first_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="folder")
            requests_after_first_download = (
                stub.shared_data.download_requests,
                stub.shared_data.tree_requests,
            )

            second_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="folder")

            self.assertEqual(first_path, second_path)
            self.assertEqual(
                requests_after_first_download,
                (stub.shared_data.download_requests, stub.shared_data.tree_requests),
            )

    def test_missing_folder_preserves_exact_file_404(self) -> None:
        with create_test_cache() as d:
            with self.assertRaises(KaggleApiHTTPError) as raised:
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="missing")

            self.assertEqual(404, raised.exception.response.status_code)
            self.assertFalse(os.path.exists(os.path.join(d, EXPECTED_DATASET_SUBPATH, "missing")))

    def test_empty_folder_preserves_exact_file_404(self) -> None:
        with create_test_cache() as d:
            with self.assertRaises(KaggleApiHTTPError) as raised:
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="empty")

            self.assertEqual(404, raised.exception.response.status_code)
            self.assertFalse(os.path.exists(os.path.join(d, EXPECTED_DATASET_SUBPATH, "empty")))

    def test_folder_download_with_output_dir(self) -> None:
        with create_test_cache(), TemporaryDirectory() as dest:
            folder_path = kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                path="folder",
                output_dir=dest,
            )

            self.assertEqual(os.path.join(dest, "folder"), folder_path)
            self.assertTrue(os.path.isfile(os.path.join(folder_path, "nested", "child.txt")))

    def test_incomplete_output_folder_requires_force(self) -> None:
        with create_test_cache(), TemporaryDirectory() as dest:
            folder_path = os.path.join(dest, "folder")
            os.makedirs(folder_path)
            with open(os.path.join(folder_path, "stale.txt"), "w") as output_file:
                output_file.write("stale")

            with self.assertRaises(FileExistsError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="folder", output_dir=dest)

    def test_force_download_replaces_folder(self) -> None:
        with create_test_cache(), TemporaryDirectory() as dest:
            folder_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="folder", output_dir=dest)
            stale_file = os.path.join(folder_path, "stale.txt")
            with open(stale_file, "w") as output_file:
                output_file.write("stale")

            refreshed_path = kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                path="folder",
                output_dir=dest,
                force_download=True,
            )

            self.assertEqual(folder_path, refreshed_path)
            self.assertFalse(os.path.exists(stale_file))
            self.assertTrue(os.path.isfile(os.path.join(folder_path, "root.txt")))

    def test_partial_folder_failure_is_resumable_in_default_cache(self) -> None:
        with create_test_cache() as d:
            stub.shared_data.fail_once_paths = {"partial/second.txt"}
            with self.assertRaises(KaggleApiHTTPError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="partial")

            h = parse_dataset_handle(VERSIONED_DATASET_HANDLE)
            cache = Cache()
            self.assertIsNone(cache.load_from_cache(h, "partial"))
            self.assertTrue(os.path.isfile(os.path.join(d, EXPECTED_DATASET_SUBPATH, "partial", "first.txt")))
            requests_before_retry = stub.shared_data.download_requests

            folder_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="partial")

            self.assertTrue(os.path.isfile(os.path.join(folder_path, "second.txt")))
            self.assertEqual(2, stub.shared_data.download_requests - requests_before_retry)

    def test_user_path_traversal_is_rejected_before_network_access(self) -> None:
        with create_test_cache():
            unsafe_paths = (
                "../escape.txt",
                "folder/../../escape.txt",
                "/absolute/path",
                "\\absolute\\path",
                "C:\\absolute\\path",
                "//server/share/path",
                "folder/evil\x00name",
            )
            for unsafe_path in unsafe_paths:
                with self.subTest(unsafe_path), self.assertRaises(ValueError):
                    kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path=unsafe_path)

        self.assertEqual(0, stub.shared_data.get_dataset_requests)
        self.assertEqual(0, stub.shared_data.download_requests)
        self.assertEqual(0, stub.shared_data.tree_requests)

    def test_server_path_traversal_is_rejected_before_writing(self) -> None:
        with create_test_cache() as d:
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="malicious")

            self.assertFalse(os.path.exists(os.path.join(d, EXPECTED_DATASET_SUBPATH, "escape.txt")))

    def test_repeated_tree_page_token_fails_closed(self) -> None:
        with create_test_cache():
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path="loop")

    def test_versioned_dataset_download_with_auto_compressed_path(self) -> None:
        with create_test_cache() as d:
            self._download_test_file_and_assert_downloaded_auto_compressed(d, VERSIONED_DATASET_HANDLE)

    def test_versioned_dataset_download_with_output_dir(self) -> None:
        with create_test_cache():
            with TemporaryDirectory() as dest:
                dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, output_dir=dest)
                self.assertEqual(dest, dataset_path)
                self.assertEqual([".complete", "foo.txt"], sorted(os.listdir(dest)))

    def test_versioned_dataset_download_with_path_and_output_dir(self) -> None:
        with create_test_cache():
            with TemporaryDirectory() as dest:
                dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path=TEST_FILEPATH, output_dir=dest)
                self.assertEqual(os.path.join(dest, TEST_FILEPATH), dataset_path)
                with open(dataset_path) as dataset_file:
                    self.assertEqual(TEST_CONTENTS, dataset_file.read())

    def test_versioned_dataset_download_with_path_and_output_dir_existing_file_fails(self) -> None:
        with TemporaryDirectory() as dest:
            dest_file = os.path.join(dest, TEST_FILEPATH)
            with open(dest_file, "w") as output_file:
                output_file.write("old")
            with self.assertRaises(FileExistsError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, path=TEST_FILEPATH, output_dir=dest)

    def test_versioned_dataset_download_with_output_dir_existing_dir_fails(self) -> None:
        with TemporaryDirectory() as dest:
            with open(os.path.join(dest, "old.txt"), "w") as output_file:
                output_file.write("old")
            with self.assertRaises(FileExistsError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, output_dir=dest)

    def test_versioned_dataset_download_with_output_dir_overwrite(self) -> None:
        with TemporaryDirectory() as dest:
            # Download it and ensure completion marker is set.
            kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                output_dir=dest,
            )
            # Add a random file to ensure the directory is overriden.
            with open(os.path.join(dest, "old.txt"), "w") as output_file:
                output_file.write("old")
            dataset_path = kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                output_dir=dest,
                force_download=True,
            )
            self.assertEqual(dest, dataset_path)
            self.assertEqual([".complete", "foo.txt"], sorted(os.listdir(dest)))

    def test_versioned_dataset_download_with_path_and_output_dir_overwrite(self) -> None:
        with TemporaryDirectory() as dest:
            # Download it and ensure completion marker is set.
            kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                path=TEST_FILEPATH,
                output_dir=dest,
            )
            # Update the file to ensure overwrite works.
            dest_file = os.path.join(dest, TEST_FILEPATH)
            with open(dest_file, "w") as output_file:
                output_file.write("old")
            dataset_path = kagglehub.dataset_download(
                VERSIONED_DATASET_HANDLE,
                path=TEST_FILEPATH,
                output_dir=dest,
                force_download=True,
            )
            self.assertEqual(dest_file, dataset_path)
            with open(dataset_path) as dataset_file:
                self.assertEqual(TEST_CONTENTS, dataset_file.read())

    def test_unversioned_dataset_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR, force_download=True
            )

    def test_versioned_dataset_download_with_force_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(
                d, VERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR, force_download=True
            )

    @parameterized(VERSIONED_DATASET_HANDLE, UNVERSIONED_DATASET_HANDLE)
    def test_dataset_full_download_with_file_already_cached(self, dataset_handle: str) -> None:
        with create_test_cache() as d:
            # Download a single file first
            kagglehub.dataset_download(dataset_handle, path=TEST_FILEPATH)
            self._download_dataset_and_assert_downloaded(d, dataset_handle, EXPECTED_DATASET_SUBDIR)
