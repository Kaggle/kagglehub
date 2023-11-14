import logging
from unittest import mock

import kagglehub
from kagglehub.config import get_kaggle_credentials
from tests.fixtures import BaseTestCase

logger = logging.getLogger(__name__)


class TestAuth(BaseTestCase):
    def test_login_updates_global_credentials(self):
        # Simulate user input for credentials
        with mock.patch("builtins.input") as mock_input:
            mock_input.side_effect = ["lastplacelarry", "some-key"]
            kagglehub.login()

        # Verify that the global variable contains the updated credentials
        self.assertEqual("lastplacelarry", get_kaggle_credentials().username)
        self.assertEqual("some-key", get_kaggle_credentials().key)
