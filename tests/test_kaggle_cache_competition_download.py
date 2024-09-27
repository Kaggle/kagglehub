import os
from unittest import mock

import requests

import kagglehub
from kagglehub.config import DISABLE_KAGGLE_CACHE_ENV_VAR_NAME
from kagglehub.env import KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import jwt_stub as stub
from .server_stubs import serv

INVALID_ARCHIVE_COMPETITION_HANDLE = "invalid/invalid"
COMPETITION_HANDLE = "squid-game"
TEST_FILEPATH = "foo.txt"


# Test cases for the CompetitionKaggleCacheResolver.
class TestKaggleCacheCompetitionDownload(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app, KAGGLE_DATA_PROXY_URL_ENV_VAR_NAME, "http://localhost:7778")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_kaggle_resolver_skipped(self) -> None:
        with mock.patch.dict(os.environ, {DISABLE_KAGGLE_CACHE_ENV_VAR_NAME: "true"}):
            with stub.create_env():
                # Assert that a ConnectionError is set (uses HTTP server which is not set)
                with self.assertRaises(requests.exceptions.ConnectionError):
                    kagglehub.competition_download(COMPETITION_HANDLE)

    def test_competition_download(self) -> None:
        with stub.create_env():
            competition_path = kagglehub.competition_download(COMPETITION_HANDLE)
            self.assertEqual(["bar.csv", "foo.txt"], sorted(os.listdir(competition_path)))

    def test_competition_download_with_path(self) -> None:
        with stub.create_env():
            competition_path = kagglehub.competition_download(COMPETITION_HANDLE, "bar.csv")
            self.assertTrue(competition_path.endswith("bar.csv"))
            self.assertTrue(os.path.isfile(competition_path))

    def test_competition_download_with_missing_file_raises(self) -> None:
        with stub.create_env():
            with self.assertRaises(ValueError):
                kagglehub.competition_download(COMPETITION_HANDLE, "missing.txt")

    def test_competition_download_bad_handle_raises(self) -> None:
        with self.assertRaises(ValueError):
            kagglehub.competition_download(INVALID_ARCHIVE_COMPETITION_HANDLE)

    def test_competition_download_with_force_download(self) -> None:
        with stub.create_env():
            competition_path = kagglehub.competition_download(COMPETITION_HANDLE)
            competition_path_forced = kagglehub.competition_download(COMPETITION_HANDLE, force_download=True)
            self.assertEqual(["bar.csv", "foo.txt"], sorted(os.listdir(competition_path_forced)))
            self.assertEqual(competition_path, competition_path_forced)

    def test_competition_download_with_force_download_explicitly_false(self) -> None:
        with stub.create_env():
            competition_path = kagglehub.competition_download(COMPETITION_HANDLE, force_download=False)
            self.assertEqual(["bar.csv", "foo.txt"], sorted(os.listdir(competition_path)))
