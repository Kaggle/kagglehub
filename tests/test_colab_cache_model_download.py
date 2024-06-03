import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_COLAB_CACHE_ENV_VAR_NAME, TBE_RUNTIME_ADDR_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import colab_stub as stub
from .server_stubs import serv

VERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b/1"
LATEST_MODEL_VERSION = 2
UNVERSIONED_MODEL_HANDLE = "metaresearch/llama-2/pyTorch/13b"
TEST_FILEPATH = "config.json"
UNAVAILABLE_MODEL_HANDLE = "unavailable/model/handle/colab/1"


class TestColabCacheModelDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        # Important, to match Colab TBE_RUNTIME_ADDR value in production, we don't prepend `http://`.
        # The `http://` is prepended inside the ColabClient class.
        cls.server = serv.start_server(stub.app, TBE_RUNTIME_ADDR_ENV_VAR_NAME, "localhost:7779")

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

    def test_colab_resolver_skipped_when_disable_colab_cache_env_var_name(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_COLAB_CACHE_ENV_VAR_NAME: "true"}):
            with stub.create_env():
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.model_download(VERSIONED_MODEL_HANDLE)

    def test_versioned_model_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.model_download("bad handle")


class TestNoInternetColabCacheModelDownload(BaseTestCase):
    def test_colab_resolver_skipped_when_model_not_present(self) -> None:
        with stub.create_env():
            # Assert that a ConnectionError is set (uses HTTP server which is not set)
            with self.assertRaises(requests.exceptions.ConnectionError):
                kagglehub.model_download(UNAVAILABLE_MODEL_HANDLE)
