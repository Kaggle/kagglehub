import unittest

from kagglehub.config import clear_kaggle_credentials


class BaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the global variable before each test
        clear_kaggle_credentials()

    def tearDown(self) -> None:
        # Reset the global variable after each test
        clear_kaggle_credentials()
