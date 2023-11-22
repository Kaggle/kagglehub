import base64
import http.server
import io
import json
import logging
from unittest import mock

import kagglehub
from kagglehub.auth import _capture_logger_output, logger
from kagglehub.config import get_kaggle_credentials
from tests.fixtures import BaseTestCase

from .utils import create_test_http_server

GOOD_CREDENTIALS_USERNAME = "lastplacelarry"
GOOD_CREDENTIALS_API_KEY = "some-key"


class KaggleAPIHandler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):  # noqa: N802
        self.send_response(200)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/v1/hello":
            # Get the basic auth credentials attached to the request
            credentials = self.headers.get("Authorization", "").split(" ")[1]
            username, key = base64.b64decode(credentials.encode("utf-8")).decode("utf-8").split(":")

            # Compare to the expected good credentials username/key
            if username == GOOD_CREDENTIALS_USERNAME and key == GOOD_CREDENTIALS_API_KEY:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(bytes(json.dumps({"message": "Hello from test server!"}), "utf-8"))
            else:
                self.send_response(403)
                self.send_header("Content-type", "application/text")
                self.send_header("Content-Length", 0)
                self.end_headers()
                self.wfile.write(bytes("", "utf-8"))
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

    def test_login_returns_403_for_bad_credentials(self):
        output_stream = io.StringIO()
        handler = logging.StreamHandler(output_stream)
        logger.addHandler(handler)
        with create_test_http_server(KaggleAPIHandler):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["invalid", "invalid"]
                kagglehub.login()

            captured_output = output_stream.getvalue()
            self.assertEqual(
                captured_output,
                "Invalid Kaggle credentials. You can check your credentials on the [Kaggle settings page](https://www.kaggle.com/settings/account).\n",
            )

    def test_capture_logger_output(self):
        with _capture_logger_output() as output:
            logger.info("This is an info message")
            logger.error("This is an error message")

        self.assertEqual(output.getvalue(), "This is an info message\nThis is an error message\n")
