import os

import kagglehub
from kagglehub.cache import DATASETS_CACHE_SUBFOLDER, get_cached_archive_path
from tests.fixtures import BaseTestCase

from .server_stubs import dataset_download_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

INVALID_ARCHIVE_DATASET_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_DATASET_HANDLE = "sarahjeffreson/large-random-spotify-artist-sample-with-metadata/versions/1"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/large-random-spotify-artist-sample-with-metadata"
TEST_FILEPATH = "CLEANED_Spotify_artist_info_Mnth-Lstnrs.csv"
TEST_CONTENTS = "{}"

EXPECTED_DATASET_SUBDIR = os.path.join(DATASETS_CACHE_SUBFOLDER, "sarahjeffreson", "large-random-spotify-artist-sample-with-metadata", "1")
EXPECTED_DATASET_SUBPATH = os.path.join(
    DATASETS_CACHE_SUBFOLDER,
    "sarahjeffreson",
    "large-random-spotify-artist-sample-with-metadata",
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
        dataset_path = kagglehub.dataset_download(dataset_handle)

        print(dataset_path)

        self.assertEqual(os.path.join(d, expected_subdir_or_subpath), dataset_path)
        self.assertEqual(["CLEANED_Spotify_artist_info_Mnth-Lstnrs.csv", "dataset"], sorted(os.listdir(dataset_path)))

        # Assert that the archive file has been deleted
        archive_path = get_cached_archive_path(dataset_handle)
        self.assertFalse(os.path.exists(archive_path))

    # def _download_test_file_and_assert_downloaded(self, d: str, dataset_handle: str, **kwargs) -> None:  # noqa: ANN003
    #     dataset_path = kagglehub.dataset_download(dataset_handle, path=TEST_FILEPATH, **kwargs)
    #     self.assertEqual(os.path.join(d, EXPECTED_DATASET_SUBPATH), dataset_path)
    #     with open(dataset_path) as dataset_file:
    #         self.assertEqual(TEST_CONTENTS, dataset_file.readline())

    def test_unversioned_dataset_download(self) -> None:
        with create_test_cache() as d:
            self._download_dataset_and_assert_downloaded(d, UNVERSIONED_DATASET_HANDLE, EXPECTED_DATASET_SUBDIR)
