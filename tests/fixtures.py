import unittest
from kagglehub.config import set_kaggle_credentials

class BaseTest(unittest.TestCase):
    def setUp(self):
        # Reset the global variable before each test
        set_kaggle_credentials(None, None)

    def tearDown(self):
        # Reset the global variable after each test
        set_kaggle_credentials(None, None)