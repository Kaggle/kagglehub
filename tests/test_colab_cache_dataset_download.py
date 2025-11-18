import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_COLAB_CACHE_ENV_VAR_NAME, TBE_RUNTIME_ADDR_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import colab_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

VERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata/versions/1"
UNVERSIONED_DATASET_HANDLE = "sarahjeffreson/featured-spotify-artiststracks-with-metadata"
TEST_FILEPATH = "foo.txt"
TEST_CONTENTS = "foo\n"
UNAVAILABLE_DATASET_HANDLE = "unavailable/dataset/versions/1"


class TestColabCacheDatasetDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        # Important, to match Colab TBE_RUNTIME_ADDR value in production, we don't prepend `http://`.
        # The `http://` is prepended inside the ColabClient class.
        cls.server = serv.start_server(stub.app, TBE_RUNTIME_ADDR_ENV_VAR_NAME, "localhost:7779")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_unversioned_dataset_download(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(UNVERSIONED_DATASET_HANDLE)
            self.assertTrue(dataset_path.endswith("/2"))
            self.assertEqual(["bar.csv", "foo.txt"], sorted(os.listdir(dataset_path)))

    def test_versioned_dataset_download(self) -> None:
        with stub.create_env():
            dataset_path = kagglehub.dataset_download(VERSIONED_DATASET_HANDLE)
            self.assertTrue(dataset_path.endswith("/1"))
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

    def test_colab_resolver_skipped_when_disable_colab_cache_env_var_name(self) -> None:
        with create_test_cache():  # falls back to http resolver, make sure to use a test cache.
            with mock.patch.dict(os.environ, {DISABLE_COLAB_CACHE_ENV_VAR_NAME: "true"}):
                with stub.create_env():
                    # Assert that a ConnectionError is set (uses HTTP server which is not set)
                    with self.assertRaises(requests.exceptions.ConnectionError):
                        kagglehub.dataset_download(VERSIONED_DATASET_HANDLE)

    def test_versioned_dataset_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.dataset_download("bad handle")


class TestNoInternetColabCacheModelDownload(BaseTestCase):
    def test_colab_resolver_skipped_when_dataset_not_present(self) -> None:
        with stub.create_env():
            # Assert that a ConnectionError is set (uses HTTP server which is not set)
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.dataset_download(UNAVAILABLE_DATASET_HANDLE)
