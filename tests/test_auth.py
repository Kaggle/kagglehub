import io
import json
import unittest
from unittest import mock
from unittest.mock import patch

import kagglehub
from kagglehub.config import CREDENTIALS_JSON_KEY, CREDENTIALS_JSON_USERNAME, _get_kaggle_credentials_file, get_kaggle_credentials


class TestAuth(unittest.TestCase):
    def test_login_prompts_for_username(self):
        # Simulate user input for username
        with mock.patch('builtins.input', return_value='ABC'):
            kagglehub.login()

        # Verify that the global variable is updated with the provided username
        self.assertEqual('ABC', get_kaggle_credentials.username)

    def test_login_prompts_for_api_key(self):
        # Simulate user input for API key
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['some-username', 'some-key']
            kagglehub.login()

        # Verify that the global variable is updated with the provided API key
        self.assertEqual('some-key', get_kaggle_credentials.key)

    def test_login_saves_credentials_to_file(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['ABC', 'some-key']
            kagglehub.login()

        # Verify that the credentials are saved to the expected file
        creds_filepath = _get_kaggle_credentials_file()
        with open(creds_filepath) as creds_json_file:
            creds_dict = json.load(creds_json_file)

        self.assertEqual('ABC', creds_dict[CREDENTIALS_JSON_USERNAME])
        self.assertEqual('some-key', creds_dict[CREDENTIALS_JSON_KEY])

    def test_login_prints_success_message(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['ABC', 'some-key']
            with mock.capture_output() as captured:
                kagglehub.login()

            # Verify that the success message is printed
            self.assertRegex(captured.getvalue(), r"You are now logged in to Kaggle Hub\.")
