import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests

from kagglehub.clients import KaggleTokenAuth
from kagglehub.env import KAGGLE_NOTEBOOK_ENV_VAR_NAME, KAGGLE_TOKEN_KEY_DIR_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import kaggle_token_auth_stub as stub
from .server_stubs import serv


class TestKaggleTokenClient(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        d = TemporaryDirectory()
        cls.d = d
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.d.cleanup()
        cls.server.shutdown()

    def test_create_model(self) -> None:
        with TemporaryDirectory() as d:
            with mock.patch.dict(
                os.environ,
                {
                    KAGGLE_NOTEBOOK_ENV_VAR_NAME: "test",
                    KAGGLE_TOKEN_KEY_DIR_ENV_VAR_NAME: f"{d}/etc/secrets/kaggle/api-v1-token",
                },
            ):
                host = os.environ["KAGGLE_API_ENDPOINT"]
                path = os.environ[KAGGLE_TOKEN_KEY_DIR_ENV_VAR_NAME]
                _ = requests.post(f"{host}/setup/{path}")
                resp = requests.get(f"{host}/api/v1/hello", auth=KaggleTokenAuth(), timeout=60)
                self.assertIsNotNone(resp)
