import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.env import KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import jwt_stub as stub
from .server_stubs import serv

INVALID_ARCHIVE_DATASET_HANDLE = "invalid/invalid/invalid/invalid/invalid"
VERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata/versions/1"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
LATEST_DATASET_VERSION = 2
TEST_FILEPATH = "foo.txt"


# Test cases for the DatasetKaggleCacheResolver.
class TestKaggleCacheDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, "http://localhost:7778")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_unversioned_dataset_download(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(UNVERSIONED_DATASET_HANDLE)
            self.assertEqual(["bar.csv", "foo.txt"], sorted(os.listdir(dataset_path)))

    def test_versioned_dataset_download(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE)
            self.assertEqual(["foo.txt"], sorted(os.listdir(dataset_path)))

    def test_versioned_dataset_download_with_path(self) -> None:
        with stub.create_env():
            dataset_file_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, "foo.txt")
            self.assertTrue(dataset_file_path.endswith("foo.txt"))
            self.assertTrue(os.path.isfile(dataset_file_path))

    def test_unversioned_dataset_download_with_path(self) -> None:
        with stub.create_env():
            dataset_file_path = kagglehub.dataset_download(UNVERSIONED_DATASET_HANDLE, "bar.csv")
            self.assertTrue(dataset_file_path.endswith("bar.csv"))
            self.assertTrue(os.path.isfile(dataset_file_path))

    def test_versioned_dataset_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, "missing.txt")

    def test_unversioned_dataset_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.dataset_download(UNVERSIONED_DATASET_HANDLE, "missing.txt")

    def test_kaggle_resolver_skipped(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_KAGGLE_CACHE_ENV_VAR_NAME: "true"}):
            with stub.create_env():
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.dataset_download(VERSIONED_DATASET_HANDLE)

    def test_versioned_dataset_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.dataset_download("bad handle")

    def test_versioned_dataset_download_with_force_download(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE)
            dataset_path_forced = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, force_download=True)

            # Using force_download shouldn't change the expected output of model_download.
            self.assertEqual(["foo.txt"], sorted(os.listdir(dataset_path_forced)))
            self.assertEqual(dataset_path, dataset_path_forced)

    def test_versioned_dataset_download_with_force_download_explicitly_false(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE, force_download=False)
            self.assertEqual(["foo.txt"], sorted(os.listdir(dataset_path)))
