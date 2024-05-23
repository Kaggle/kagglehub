import os

import kagglehub
from kagglehub.cache import DATASETS_CACHE_SUBFOLDER, get_cached_archive_path
from kagglehub.handle import parse_dataset_handle
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

INVALID_ARCHIVE_DATASET_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata/versions/1"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo"

EXPECTED_DATASET_SUBDIR = os.path.join(DATASETS_CACHE_SUBFOLDER, "sarahjeffreson", "featured-spotify-artiststracks-with-metadata", "1")
EXPECTED_DATASET_SUBPATH = os.path.join(
    DATASETS_CACHE_SUBFOLDER,
    "sarahjeffreson",
    "featured-spotify-artiststracks-with-metadata",
    "1",
    TEST_FILEPATH,
)

class TestHttpDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):  # noqa: ANN102
        serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):  # noqa: ANN102
        serv.stop_server()

    def _download_dataset_and_assert_downloaded(
        self,
        d: str,
        dataset_handle: str,
        expected_subdir_or_subpath: str,
        **kwargs,  # noqa: ANN003
    ) -> None:
        # Download the full datasets and ensure all files are there.        
        dataset_path = kagglehub.dataset_download(dataset_handle, **kwargs)

        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), dataset_path)
        self.assertEqual(["foo.txt", "dataset"], sorted(os.listdir(dataset_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(parse_dataset_handle(dataset_handle))
        self.assertFalse(os.path.exists(archive_path))

    # def _download_test_file_and_assert_downloaded(self, d: str, dataset_handle: str, **kwargs) -> None:  # noqa: ANN003
    #     dataset_path = kagglehub.dataset_download(dataset_handle, path=TEST_FILEPATH, **kwargs)
    #     self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH), dataset_path)
    #     with open(dataset_path) as dataset_file:
    #         self.assertEqual(TEST_CONTENTS, dataset_file.readline())

    def test_unversioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)

    # def test_versioned_model_download_bad_archive(self) -> None:
    #     with create_test_cache():
    #         with self.assertRaises(ValueError):
    #             kagglehub.model_download(INVALID_ARCHIVE_DATASET_HANDLE)
