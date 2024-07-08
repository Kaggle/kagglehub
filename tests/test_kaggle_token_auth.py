import os
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from tempfile import TemporaryDirectory
from unittest import mock

import jwt
import requests

from kagglehub.clients import KaggleTokenAuth
from kagglehub.env import KAGGLE_NOTEBOOK_ENV_VAR_NAME, KAGGLE_TOKEN_KEY_DIR_ENV_VAR_NAME
from tests.fixtures import BaseTestCase

from .server_stubs import kaggle_token_auth_stub as stub
from .server_stubs import serv


class MockInvalidTokenAuth(requests.auth.AuthBase):
    def __init__(self):
        payload = {"user_id": "lastplacelarry", "exp": datetime.now(timezone.utc) + timedelta(days=1)}
        self.token = jwt.encode(payload, "super secret key", algorithm="HS256")

    def __call__(self, r: requests.PreparedRequest):
        r.headers["Authorization"] = f"Bearer {self.token}"
        return r


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
                r = requests.post(f"{host}/setup/{path}", timeout=60)
                self.assertEqual(r.status_code, HTTPStatus.OK)

                resp = requests.get(f"{host}/api/v1/hello", auth=KaggleTokenAuth(), timeout=60)
                self.assertEqual(resp.status_code, HTTPStatus.OK)
                self.assertIsNotNone(resp)

    def test_bad_auth_fails(self) -> None:
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
                r = requests.post(f"{host}/setup/{path}", timeout=60)
                self.assertEqual(r.status_code, HTTPStatus.OK)

                resp = requests.get(f"{host}/api/v1/hello", auth=MockInvalidTokenAuth(), timeout=60)
                self.assertGreaterEqual(resp.status_code, 400)
