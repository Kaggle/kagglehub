import unittest

class BaseTest(unittest.TestCase):
    def setUp(self):
        # Reset the global variable before each test
        global _kaggle_credentials
        _kaggle_credentials = None

    def tearDown(self):
        # Reset the global variable after each test
        global _kaggle_credentials
        _kaggle_credentials = None