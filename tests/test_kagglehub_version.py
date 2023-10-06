import unittest

import kagglehub


class TestKagglehubVersion(unittest.TestCase):
    def test_version_is_set(self):
        self.assertIsNotNone(kagglehub.__version__)
