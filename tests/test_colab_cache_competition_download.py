import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_COLAB_CACHE_ENV_VAR_NAME, TBE_RUNTIME_ADDR_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import colab_stub as stub
from .server_stubs import serv
from .utils import create_test_cache

COMPETITION_HANDLE = "titanic"
UNAVAILABLE_COMPETITION_HANDLE = "unavailable-competition"


class TestColabCacheCompetitionDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        # Important, to match Colab TBE_RUNTIME_ADDR value in production, we don't prepend `http://`.
        # The `http://` is prepended inside the ColabClient class.
        cls.server = serv.start_server(stub.app, TBE_RUNTIME_ADDR_ENV_VAR_NAME, "localhost:7780")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_competition_download(self) -> None:
        with stub.create_env():
            competition_path = kagglehub.competition_download(COMPETITION_HANDLE)
            self.assertTrue(competition_path.endswith("/competitions/titanic"))
            self.assertEqual(["competition_data.csv"], sorted(os.listdir(competition_path)))

    def test_competition_download_with_path(self) -> None:
        with stub.create_env():
            competition_file_path = kagglehub.competition_download(COMPETITION_HANDLE, "competition_data.csv")
            self.assertTrue(competition_file_path.endswith("competition_data.csv"))
            self.assertTrue(os.path.isfile(competition_file_path))

    def test_competition_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.competition_download(COMPETITION_HANDLE, "missing.csv")

    def test_colab_resolver_skipped_when_disable_colab_cache_env_var_name(self) -> None:
        with create_test_cache():  # falls back to http resolver, make sure to use a test cache.
            with mock.patch.dict(os.environ, {DISABLE_COLAB_CACHE_ENV_VAR_NAME: "true"}):
                with stub.create_env():
                    # Assert that a UnauthenticatedError is raised because we fall back to HTTP resolver which requires auth
                    with self.assertRaises(kagglehub.exceptions.UnauthenticatedError):
                        kagglehub.competition_download(COMPETITION_HANDLE)


class TestNoInternetColabCacheCompetitionDownload(BaseTestCase):
    def test_colab_resolver_skipped_when_competition_not_present(self) -> None:
        with stub.create_env():
            # Assert that a UnauthenticatedError is raised because we fall back to HTTP resolver which requires auth
            with self.assertRaises(kagglehub.exceptions.UnauthenticatedError):
                kagglehub.competition_download(UNAVAILABLE_COMPETITION_HANDLE)
