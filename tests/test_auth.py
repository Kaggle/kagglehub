import unittest

import kagglehub


class TestAuth(unittest.TestCase):
    def test_login(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.login()
