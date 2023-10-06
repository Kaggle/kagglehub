import unittest

import kagglehub


class TestModelDownload(unittest.TestCase):
    def test_model_download(self):
        with self.assertRaises(NotImplementedError):
            kagglehub.model_download()
