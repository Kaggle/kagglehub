import io
import logging
from unittest import mock

import kagglehub
from kagglehub.auth import _capture_logger_output
from kagglehub.auth import _logger as logger
from kagglehub.config import get_kaggle_credentials
from kagglehub.exceptions import UnauthenticatedError
from tests.fixtures import BaseTestCase

from .server_stubs import auth_stub as stub
from .server_stubs import serv


class TestAuth(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serv.start_server(stub.app)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_login_updates_global_credentials(self) -> None:
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["lastplacelarry", "some-key"]
            kagglehub.login()

        # Verify that the global variable contains the updated credentials
        credentials = get_kaggle_credentials()
        if credentials is None:
            self.fail("Credentials should not be None")
        self.assertEqual("lastplacelarry", credentials.username)
        self.assertEqual("some-key", credentials.key)

    def test_login_updates_global_credentials_no_validation(self) -> None:
        # Simulate user input for credentials
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["lastplacelarry", "some-key"]
            kagglehub.login(validate_credentials=False)

        # Verify that the global variable contains the updated credentials
        credentials = get_kaggle_credentials()
        if credentials is None:
            self.fail("Credentials should not be None")
        self.assertEqual("lastplacelarry", credentials.username)
        self.assertEqual("some-key", credentials.key)

    def test_set_kaggle_credentials_raises_error_with_empty_username(self) -> None:
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["", "some-key"]
                kagglehub.login()

    def test_set_kaggle_credentials_raises_error_with_empty_api_key(self) -> None:
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["lastplacelarry", ""]
                kagglehub.login()

    def test_set_kaggle_credentials_raises_error_with_empty_username_api_key(self) -> None:
        with self.assertRaises(ValueError):
            with mock.patch("builtins.input") as mock_input:
                mock_input.side_effect = ["", ""]
                kagglehub.login()

    def test_login_returns_403_for_bad_credentials(self) -> None:
        output_stream = io.StringIO()
        handler = logging.StreamHandler(output_stream)
        logger.addHandler(handler)
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["invalid", "invalid"]
            kagglehub.login()

        captured_output = output_stream.getvalue()
        self.assertEqual(
            captured_output,
            "Invalid Kaggle credentials. You can check your credentials on the [Kaggle settings page](https://www.kaggle.com/settings/account).\n",
        )

    def test_capture_logger_output(self) -> None:
        with _capture_logger_output() as output:
            logger.info("This is an info message")
            logger.error("This is an error message")

        self.assertEqual(output.getvalue(), "This is an info message\nThis is an error message\n")

    def test_whoami_raises_unauthenticated_error(self) -> None:
        with self.assertRaises(UnauthenticatedError):
            kagglehub.whoami()

    def test_whoami_success(self) -> None:
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["lastplacelarry", "some-key"]
            kagglehub.login(validate_credentials=False)
            result = kagglehub.whoami()

            self.assertEqual(result, {"username": "lastplacelarry"})
