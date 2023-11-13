import io
import json
import unittest
import logging
from unittest import mock
from unittest.mock import patch
import sys
import kagglehub
from kagglehub.config import get_kaggle_credentials
from tests.fixtures import BaseTestCase

logger = logging.getLogger(__name__)


class TestAuth(BaseTestCase):

    def test_login_updates_global_credentials(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['lastplacelarry', 'some-key']
            kagglehub.login()

        # Verify that the global variable contains the updated credentials
        self.assertEqual('lastplacelarry', get_kaggle_credentials().username)
        self.assertEqual('some-key', get_kaggle_credentials().key)

    def test_login_prints_success_message(self):
        # Simulate user input for credentials
        with mock.patch('builtins.input') as mock_input:
            mock_input.side_effect = ['lastplacelarry', 'some-key']

            # Capture log messages using a caplog fixture
            caplog = mock.Mock()
            logger.addFilter(caplog)

            kagglehub.login()

            # Verify that the success message is logged
            last_message = caplog.pop()
            self.assertTrue(last_message.msg.startswith("You are now logged in to Kaggle Hub."))