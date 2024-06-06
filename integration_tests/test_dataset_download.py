import unittest

from requests import HTTPError

from kagglehub import dataset_download

from .utils import assert_files, create_test_cache

HANDLE = "ryanholbrook/dl-course-data/versions/5"


class TestDatasetDownload(unittest.TestCase):
    def test_dataset_versioned_succeeds(self) -> None:
        with create_test_cache():
            actual_path = dataset_download(HANDLE)

            expected_files = [
                "abalone.csv",
                "candy.csv",
                "cereal.csv",
                "concrete.csv",
                "diamonds.csv",
                "forestfires.csv",
                "fuel.csv",
                "hotel.csv",
                "housing.csv",
                "ion.csv",
                "red-wine.csv",
                "songs.csv",
                "spotify.csv",
            ]
            assert_files(self, actual_path, expected_files)

    def test_dataset_unversioned_succeeds(self) -> None:
        with create_test_cache():
            unversioned_handle = "ryanholbrook/dl-course-data"
            actual_path = dataset_download(unversioned_handle)

            expected_files = [
                "abalone.csv",
                "candy.csv",
                "cereal.csv",
                "concrete.csv",
                "diamonds.csv",
                "forestfires.csv",
                "fuel.csv",
                "hotel.csv",
                "housing.csv",
                "ion.csv",
                "red-wine.csv",
                "songs.csv",
                "spotify.csv",
            ]
            assert_files(self, actual_path, expected_files)

    def test_download_private_dataset_succeeds(self) -> None:
        with create_test_cache():
            actual_path = dataset_download("integrationtester/kagglehub-test-private-dataset")

            expected_files = [
                "private.txt",
            ]

            assert_files(self, actual_path, expected_files)

    def test_download_multiple_files(self) -> None:
        with create_test_cache():
            file_paths = ["abalone.csv", "diamonds.csv", "red-wine.csv"]
            for p in file_paths:
                actual_path = dataset_download(HANDLE, path=p)
                assert_files(self, actual_path, [p])

    def test_download_with_incorrect_file_path(self) -> None:
        incorrect_path = "nonexistent/file/path"
        with self.assertRaises(HTTPError):
            dataset_download(HANDLE, path=incorrect_path)
