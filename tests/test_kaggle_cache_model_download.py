import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.env import KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import jwt_stub as stub
from .server_stubs import serv

INVALID_ARCHIVE_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/bad-archive-variation/1"
VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/1"
LATEST_MODEL_VERSION = 2
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"


# Test cases for the ModelKaggleCacheResolver.
class TestKaggleCacheModelDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, "http://localhost:7778")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_unversioned_model_download(self) -> None:
        with stub.create_env():
            model_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/2"))
            self.assertEqual(["config.json", "model.keras"], sorted(os.listdir(model_path)))

    def test_versioned_model_download(self) -> None:
        with stub.create_env():
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            self.assertTrue(model_path.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path)))

    def test_versioned_model_download_with_path(self) -> None:
        with stub.create_env():
            model_file_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_unversioned_model_download_with_path(self) -> None:
        with stub.create_env():
            model_file_path = kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "config.json")
            self.assertTrue(model_file_path.endswith("config.json"))
            self.assertTrue(os.path.isfile(model_file_path))

    def test_versioned_model_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.model_download(VERSIONED_MODEL_HANDLE, "missing.txt")

    def test_unversioned_model_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.model_download(UNVERSIONED_MODEL_HANDLE, "missing.txt")

    def test_kaggle_resolver_skipped(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_KAGGLE_CACHE_ENV_VAR_NAME: "true"}):
            with stub.create_env():
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.model_download(VERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.model_download("bad handle")

    def test_versioned_model_download_with_force_download(self) -> None:
        with stub.create_env():
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE)
            model_path_forced = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=True)

            # Using force_download shouldn't change the expected output of model_download.
            self.assertTrue(model_path_forced.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path_forced)))
            self.assertEqual(model_path, model_path_forced)

    def test_versioned_model_download_with_force_download_explicitly_false(self) -> None:
        with stub.create_env():
            model_path = kagglehub.model_download(VERSIONED_MODEL_HANDLE, force_download=False)
            self.assertTrue(model_path.endswith("/1"))
            self.assertEqual(["config.json"], sorted(os.listdir(model_path)))
