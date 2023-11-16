import http.server
import json
import logging
from unittest import mock

import requests

import kagglehub
from kagglehub.config import get_kaggle_credentials
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

logger = logging.getLogger(__name__)


class KaggleAPIHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
        if self.path == "/hello":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps({"message": "Hello from test server!"}), "utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/text")
            self.send_header("Content-Length", 0)
            self.end_headers()
            self.wfile.write(bytes("", "utf-8"))


class TestAuth(BaseTestCase):
    def test_login_updates_global_credentials(self):
        # Simulate user input for credentials

        with create_test_http_server(KaggleAPIHandler):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["lastplacelarry", "some-key"]
                kagglehub.login()

            # Verify that the global variable contains the updated credentials
            self.assertEqual("lastplacelarry", get_kaggle_credentials().username)
            self.assertEqual("some-key", get_kaggle_credentials().key)

    def test_login_updates_global_credentials_no_validation(self):
        # Simulate user input for credentials
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["lastplacelarry", "some-key"]
            kagglehub.login(validate_credentials=False)

        # Verify that the global variable contains the updated credentials
        self.assertEqual("lastplacelarry", get_kaggle_credentials().username)
        self.assertEqual("some-key", get_kaggle_credentials().key)

    def test_set_kaggle_credentials_raises_error_with_empty_username(self):
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["", "some-key"]
                kagglehub.login()

    def test_set_kaggle_credentials_raises_error_with_empty_api_key(self):
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["lastplacelarry", ""]
                kagglehub.login()

    def test_set_kaggle_credentials_raises_error_with_empty_username_api_key(self):
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["", ""]
                kagglehub.login()
