import io
import json
import unittest
from unittest import mock
from unittest.mock import patch
import sys

import kagglehub
from kagglehub.config import get_kaggle_credentials


class TestAuth(unittest.TestCase):

    def test_login_prompts_for_username(self):
        # Simulate user input for username
        with mock.patch('builtins.input', return_value='ABC'):
            kagglehub.login()

        # Verify that the global variable is updated with the provided username
        self.assertEqual('ABC', get_kaggle_credentials().username)

    def test_login_prompts_for_api_key(self):
        # Simulate user input for API key
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['some-username', 'some-key']
            kagglehub.login()

        # Verify that the global variable is updated with the provided API key
        self.assertEqual('some-key', get_kaggle_credentials().key)

    def test_login_updates_global_credentials(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['ABC', 'some-key']
            kagglehub.login()

        # Verify that the global variable contains the updated credentials
        self.assertEqual('ABC', get_kaggle_credentials().username)
        self.assertEqual('some-key', get_kaggle_credentials().key)

    def test_login_prints_success_message(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['ABC', 'some-key']

            # Capture output using io.StringIO
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output

            try:
                kagglehub.login()
            finally:
                # Restore original stdout
                sys.stdout = old_stdout

            # Verify that the success message is printed
            self.assertRegex(captured_output.getvalue(), r"You are now logged in to Kaggle Hub\.")
